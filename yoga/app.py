import cv2
import mediapipe as mp
import math
from flask import Flask, render_template, Response, request, redirect, url_for
import json
import pyttsx3
from threading import Thread, Lock
import time
import os
from werkzeug.utils import secure_filename
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import queue
import threading
import requests
import uuid
from datetime import datetime
import traceback

# =========================================================
# 配置常量
# =========================================================
UPLOAD_FOLDER = 'static/detect'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
SPEECH_COOLDOWN = 2  # 语音播报冷却时间(秒)

# =========================================================
# Flask应用初始化
# =========================================================
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# 确保上传目录存在
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# =========================================================
# MediaPipe姿态检测初始化 - 优化配置
# =========================================================
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# 用于静态图像处理的姿势检测器
static_pose = mp_pose.Pose(
    static_image_mode=True,       # 静态图像模式
    model_complexity=1,           # 平衡精度和速度
    smooth_landmarks=False,       # 静态图像不需要平滑
    enable_segmentation=False,    # 静态图像不需要分割
    min_detection_confidence=0.7
)

# 用于视频流处理的姿势检测器
video_pose = mp_pose.Pose(
    static_image_mode=False,      # 视频模式
    model_complexity=1,           # 从2降到1
    smooth_landmarks=True,        # 启用平滑
    enable_segmentation=False,    # 禁用分割功能，避免尺寸不一致错误
    min_detection_confidence=0.7, # 更高的检测阈值减少误报
    min_tracking_confidence=0.7   # 更高的跟踪阈值提高稳定性
)

# 添加平滑处理相关变量
landmarks_history = []
MAX_HISTORY_LENGTH = 5

# =========================================================
# 全局状态变量
# =========================================================
# 语音播放控制
speak_lock = Lock()
last_speak_time = 0

# 姿态数据管理
pose_data_queue = queue.Queue(maxsize=1)
last_frame = None
frame_lock = threading.Lock()

# 角度数据管理
latest_angles = {
    '左臂角度': 0, '右臂角度': 0,
    '左上身角度': 0, '右上身角度': 0,
    '左腿角度': 0, '右腿角度': 0
}
angles_lock = Lock()

# 识别状态控制
recognition_active = False
recognition_lock = Lock()
current_training_pose = None  # 添加当前训练姿势变量

# 视频源控制
use_camera = False  # 默认使用视频文件而非摄像头
default_video = "show_video.mp4"  # 默认视频
current_video = default_video  # 当前选择的视频
video_directory = "static/videos/"  # 视频文件目录

# 性能优化：缓存字体和状态
_cached_font = None
_last_recognition_state = (False, None)  # (is_active, current_pose)
_last_pose_match_result = None  # 缓存上次的姿势匹配结果
_last_pose_match_time = 0  # 上次姿势匹配的时间
POSE_MATCH_CACHE_DURATION = 0.5  # 姿势匹配缓存时间（秒）

# MediaPipe错误处理
_mediapipe_error_count = 0
MAX_MEDIAPIPE_ERRORS = 10  # 最大错误次数

# 动态分辨率调整 - 使用大屏默认设置
_target_display_width = 1000  # 目标显示宽度 (增加到1000)
_target_display_height = 750  # 目标显示高度 (增加到750)
_display_scale_factor = 1.0  # 显示缩放因子

def calculate_optimal_display_size(input_width, input_height, max_width=1000, max_height=750):
    """
    根据输入分辨率计算最佳显示尺寸，保持宽高比 - 大屏默认设置
    
    参数:
        input_width: 输入宽度
        input_height: 输入高度
        max_width: 最大显示宽度 (增加到1000)
        max_height: 最大显示高度 (增加到750)
        
    返回:
        (display_width, display_height, scale_factor)
    """
    if input_width <= 0 or input_height <= 0:
        return max_width, max_height, 1.0
    
    # 计算宽高比
    aspect_ratio = input_width / input_height
    
    # 计算在最大尺寸限制下的显示尺寸
    if aspect_ratio > max_width / max_height:
        # 宽度优先
        display_width = max_width
        display_height = int(max_width / aspect_ratio)
    else:
        # 高度优先
        display_height = max_height
        display_width = int(max_height * aspect_ratio)
    
    # 计算缩放因子
    scale_factor = display_width / input_width
    
    return display_width, display_height, scale_factor

def get_display_size():
    """获取当前显示尺寸"""
    return _target_display_width, _target_display_height

def update_display_size(input_width, input_height):
    """更新显示尺寸"""
    global _target_display_width, _target_display_height, _display_scale_factor
    _target_display_width, _target_display_height, _display_scale_factor = calculate_optimal_display_size(
        input_width, input_height
    )
    print(f"更新显示尺寸: {input_width}x{input_height} -> {_target_display_width}x{_target_display_height} (缩放因子: {_display_scale_factor:.2f})")

def reset_mediapipe_if_needed():
    """当MediaPipe错误过多时重置实例"""
    global _mediapipe_error_count, video_pose
    _mediapipe_error_count += 1
    
    if _mediapipe_error_count >= MAX_MEDIAPIPE_ERRORS:
        print("MediaPipe错误过多，重新初始化...")
        try:
            video_pose.close()
            video_pose = mp_pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                smooth_landmarks=True,
                enable_segmentation=False,  # 禁用分割功能
                min_detection_confidence=0.7,
                min_tracking_confidence=0.7
            )
            _mediapipe_error_count = 0
            print("MediaPipe重新初始化成功")
        except Exception as e:
            print(f"MediaPipe重新初始化失败: {e}")

# =========================================================
# 加载姿势数据
# =========================================================
def load_pose_data():
    """加载姿势数据文件，如果不存在则创建"""
    try:
        with open("poses_data.json", 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 确保数据结构正确
        if "poses" not in data:
            data["poses"] = {}
            
        # 提取姿势名称列表
        pose_names = list(data.get("poses", {}).keys())
        
        return data, pose_names
    except FileNotFoundError:
        # 如果文件不存在，创建空数据结构
        data = {"poses": {}}
        with open("poses_data.json", 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        return data, []

# 加载姿势数据
poses_data, sum_data = load_pose_data()

def get_cached_font():
    """获取缓存的字体，避免重复加载"""
    global _cached_font
    if _cached_font is None:
        try:
            _cached_font = ImageFont.truetype("simhei.ttf", 36)
        except:
            try:
                _cached_font = ImageFont.truetype("NotoSansSC-Regular.otf", 36)
            except:
                _cached_font = ImageFont.load_default()
    return _cached_font

def get_recognition_state():
    """获取识别状态，减少锁操作"""
    global _last_recognition_state
    current_time = time.time()
    
    # 只在需要时更新状态，避免频繁锁操作
    with recognition_lock:
        current_state = (recognition_active, current_training_pose)
        if current_state != _last_recognition_state:
            _last_recognition_state = current_state
    return _last_recognition_state

def reset_caches():
    """重置所有缓存状态，用于状态切换时"""
    global _last_recognition_state, _last_pose_match_result, _last_pose_match_time
    _last_recognition_state = (False, None)
    _last_pose_match_result = None
    _last_pose_match_time = 0

def optimized_match_pose(pose_name, current_angles):
    """
    优化的姿势匹配函数，使用缓存减少重复计算
    """
    global _last_pose_match_result, _last_pose_match_time
    
    current_time = time.time()
    
    # 检查缓存是否有效
    if (_last_pose_match_result is not None and 
        current_time - _last_pose_match_time < POSE_MATCH_CACHE_DURATION):
        return _last_pose_match_result
    
    # 执行匹配计算
    result = match_pose(pose_name, current_angles)
    
    # 更新缓存
    _last_pose_match_result = result
    _last_pose_match_time = current_time
    
    return result

def speak_text(text):
    """播放文字转语音，带冷却时间控制 - 优化版本"""
    global last_speak_time
    
    # 检查是否距离上次播放已经过了冷却时间
    current_time = time.time()
    with speak_lock:
        if current_time - last_speak_time < SPEECH_COOLDOWN:
            return
        last_speak_time = current_time
    
    # 优化：使用线程池或异步方式，避免频繁创建新线程
    try:
        # 使用后台线程执行语音播报，但不频繁创建新线程
        if not hasattr(speak_text, '_speech_thread') or not speak_text._speech_thread.is_alive():
            speak_text._speech_thread = Thread(target=_speak_text_worker, args=(text,), daemon=True)
            speak_text._speech_thread.start()
    except Exception as e:
        print(f"Error speaking text: {e}")

def _speak_text_worker(text):
    """语音播报工作线程"""
    try:
        engine = pyttsx3.init()
        engine.setProperty('rate', 150)
        engine.setProperty('volume', 0.9)
        engine.say(text)
        engine.runAndWait()
        engine.stop()
    except Exception as e:
        print(f"Error in speech worker: {e}")

def detectPose(image, pose, use_smooth=True):
    """
    使用MediaPipe检测图像中的人体姿势关键点，并应用平滑处理
    
    参数:
        image: 输入图像
        pose: MediaPipe姿态检测器
        use_smooth: 是否应用平滑处理
        
    返回:
        output_image: 带有关键点标注的图像
        landmarks: 平滑处理后的关键点坐标列表
    """
    # 复制输入图像
    output_image = image.copy()
    
    # 将图像从BGR转换为RGB格式
    imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 执行姿态检测
    results = pose.process(imageRGB)
    
    # 获取图像高度和宽度
    height, width, _ = image.shape
    
    # 初始化关键点列表
    landmarks = []
    
    # 检查是否检测到关键点
    if results.pose_landmarks:
        # 创建绘制规格 - 使用绿色关键点和连接线
        draw_spec = mp_drawing.DrawingSpec(
            color=(0, 255, 0),
            thickness=2,
            circle_radius=2
        )
        
        # 在输出图像上绘制姿态关键点
        mp_drawing.draw_landmarks(
            image=output_image, 
            landmark_list=results.pose_landmarks,
            connections=mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=draw_spec,
            connection_drawing_spec=draw_spec
        )
        
        # 遍历检测到的关键点
        for landmark in results.pose_landmarks.landmark:
            # 将关键点添加到列表
            landmarks.append((
                int(landmark.x * width), 
                int(landmark.y * height),
                (landmark.z * width)
            ))
        
        # 应用平滑处理
        if use_smooth:
            landmarks = smooth_landmark_data(landmarks)
    
    return output_image, landmarks

def calculateAngle(landmark1, landmark2, landmark3):
    """
    计算三个关键点之间的角度
    
    参数:
        landmark1, landmark2, landmark3: 三个关键点坐标 (x, y, z)
        
    返回:
        角度值 (0-180度)
    """
    # 获取关键点坐标
    x1, y1, _ = landmark1
    x2, y2, _ = landmark2
    x3, y3, _ = landmark3

    # 计算两个向量的2D表示（忽略z轴）
    vector1 = [x1 - x2, y1 - y2]
    vector2 = [x3 - x2, y3 - y2]
    
    # 使用atan2计算角度
    angle = math.degrees(
        math.atan2(vector2[1], vector2[0]) - 
        math.atan2(vector1[1], vector1[0])
    )
    
    # 确保角度为正
    angle = angle if angle >= 0 else angle + 360
    
    # 确保角度在0-180度范围内
    if angle > 180:
        angle = 360 - angle
        
    return angle

def calculate_all_angles(landmarks):
    """
    计算所有关键关节角度
    
    参数:
        landmarks: 关键点坐标列表
        
    返回:
        关节角度列表 [左臂, 右臂, 左上身, 右上身, 左腿, 右腿]
    """
    angles = []
    
    # 1. 计算左臂的角度 (左肩-左肘-左手腕)
    angles.append(calculateAngle(
        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
        landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
        landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
    ))
    
    # 2. 计算右臂的角度 (右肩-右肘-右手腕)
    angles.append(calculateAngle(
        landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
        landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value],
        landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
    ))
    
    # 3. 计算左上身的角度 (左肘-左肩-左臀)
    angles.append(calculateAngle(
        landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
        landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
    ))
    
    # 4. 计算右上身的角度 (右臀-右肩-右肘)
    angles.append(calculateAngle(
        landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
        landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
        landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
    ))
    
    # 5. 计算左腿的角度 (左臀-左膝-左踝)
    angles.append(calculateAngle(
        landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
        landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value],
        landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]
    ))
    
    # 6. 计算右腿的角度 (右臀-右膝-右踝)
    angles.append(calculateAngle(
        landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
        landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value],
        landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]
    ))
    
    return angles

# =========================================================
# 姿势匹配和分类 - 改进误差容忍度与权重系统
# =========================================================
def match_pose(pose_name, current_angles):
    """
    检查当前角度是否匹配指定的姿势
    
    参数:
        pose_name: 要匹配的姿势名称
        current_angles: 当前检测的角度列表
        
    返回:
        是否匹配 (True/False)
    """
    try:
        # 从poses_data中获取标准角度数据
        standard_angles = poses_data["poses"].get(pose_name, {}).get("angles", [])
        if not standard_angles:
            print(f"Error: No angle data for {pose_name}")
            return False
            
        # 创建翻转后的角度列表（用于处理镜像姿势）
        flipped_angles = [
            standard_angles[1], standard_angles[0],  # 右臂和左臂互换
            standard_angles[3], standard_angles[2],  # 右上身和左上身互换
            standard_angles[5], standard_angles[4]   # 右腿和左腿互换（修正）
        ]
        
        # 创建仅左右腿交换的角度列表（专门处理腿部混淆问题）
        leg_swapped_angles = [
            standard_angles[0], standard_angles[1],  # 保持原始臂部角度
            standard_angles[2], standard_angles[3],  # 保持原始上身角度
            standard_angles[5], standard_angles[4]   # 仅交换左右腿角度
        ]
        
        # 为不同关节设置误差容忍度 - 针对瑜伽优化
        max_errors = [25, 25, 20, 20, 25, 25]  # 降低所有关节的误差容忍度
        
        # 为不同关节设置权重 - 瑜伽中腰部和腿部姿势通常更重要
        joint_weights = [1.0, 1.0, 1.5, 1.5, 1.3, 1.3]  # 上身和腿部有更高权重
        
        # 计算原始角度的加权匹配分数
        original_score = 0
        original_matches = 0
        for i in range(len(standard_angles)):
            error = abs(standard_angles[i] - current_angles[i])
            if error <= max_errors[i]:
                original_matches += 1
                original_score += (1 - error / max_errors[i]) * joint_weights[i]
        
        # 计算翻转角度的加权匹配分数
        flipped_score = 0
        flipped_matches = 0
        for i in range(len(flipped_angles)):
            error = abs(flipped_angles[i] - current_angles[i])
            if error <= max_errors[i]:
                flipped_matches += 1
                flipped_score += (1 - error / max_errors[i]) * joint_weights[i]
        
        # 计算仅左右腿交换的匹配分数
        leg_swapped_score = 0
        leg_swapped_matches = 0
        for i in range(len(leg_swapped_angles)):
            error = abs(leg_swapped_angles[i] - current_angles[i])
            if error <= max_errors[i]:
                leg_swapped_matches += 1
                leg_swapped_score += (1 - error / max_errors[i]) * joint_weights[i]
        
        # 使用最佳匹配得分（原始、全身翻转或仅腿部翻转）
        best_score = max(original_score, flipped_score, leg_swapped_score)
        best_matches = max(original_matches, flipped_matches, leg_swapped_matches)
        
        # 用于调试的匹配方式
        match_type = "原始"
        if best_score == flipped_score and flipped_score > original_score:
            match_type = "全身翻转"
        elif best_score == leg_swapped_score and leg_swapped_score > original_score and leg_swapped_score > flipped_score:
            match_type = "仅腿部翻转"
        
        # 设置匹配阈值 - 提高要求
        min_matches = 5  # 从4提高到5，要求更多关节匹配
        min_score = sum(joint_weights) * 0.7  # 要求至少70%的加权得分(原来是60%)
        
        # 判断是否匹配
        matches = best_matches >= min_matches and best_score >= min_score
        
        # 添加调试信息
        if matches:
            print(f"姿势匹配: {pose_name} (匹配方式: {match_type})")
            print(f"当前角度: {current_angles}")
            print(f"标准角度: {standard_angles}")
            print(f"最佳匹配数: {best_matches}/6, 加权得分: {best_score:.2f}/{sum(joint_weights):.2f}")
        
        return matches
    except Exception as e:
        print(f"匹配姿势时出错: {str(e)}")
        return False

def find_best_matching_pose(current_angles):
    """
    查找最匹配当前角度的姿势
    
    参数:
        current_angles: 当前检测到的角度列表
    
    返回:
        (最佳匹配姿势名称, 匹配关节数量, 匹配得分)
    """
    best_match = None
    best_matches = 0
    best_total_score = 0
    
    # 瑜伽中腰部和腿部姿势通常更重要
    joint_weights = [1.0, 1.0, 1.5, 1.5, 1.3, 1.3]  # 上身和腿部有更高权重
    
    # 遍历所有姿势寻找最佳匹配
    for pose_name in sum_data:
        try:
            # 获取标准角度数据
            standard_angles = poses_data["poses"].get(pose_name, {}).get("angles", [])
            if not standard_angles:
                continue
                
            # 创建翻转角度列表（处理镜像姿势）
            flipped_angles = [
                standard_angles[1], standard_angles[0],
                standard_angles[3], standard_angles[2],
                360-standard_angles[5], 360-standard_angles[4]
            ]
            
            # 为不同关节设置误差容忍度 - 针对瑜伽优化
            max_errors = [25, 25, 20, 20, 25, 25]  # 降低所有关节的误差容忍度
            
            # 计算原始角度的加权匹配分数
            original_score = 0
            original_matches = 0
            for i in range(len(standard_angles)):
                error = abs(standard_angles[i] - current_angles[i])
                if error <= max_errors[i]:
                    original_matches += 1
                    original_score += (1 - error / max_errors[i]) * joint_weights[i]
            
            # 计算翻转角度的加权匹配分数
            flipped_score = 0
            flipped_matches = 0
            for i in range(len(flipped_angles)):
                error = abs(flipped_angles[i] - current_angles[i])
                if error <= max_errors[i]:
                    flipped_matches += 1
                    flipped_score += (1 - error / max_errors[i]) * joint_weights[i]
            
            # 使用最佳匹配结果
            current_matches = max(original_matches, flipped_matches)
            current_score = max(original_score, flipped_score)
            
            # 更新最佳匹配
            if current_score > best_total_score:
                best_total_score = current_score
                best_matches = current_matches
                best_match = pose_name
                
        except Exception as e:
            print(f"处理姿势 {pose_name} 时出错: {str(e)}")
    
    return best_match, best_matches, best_total_score

def classifyPose(landmarks, output_image, angles=None):
    """
    对检测到的姿势进行分类，并在图像上标注结果
    
    参数:
        landmarks: 检测到的关键点坐标
        output_image: 输出图像
        angles: 预计算的角度列表(可选)
    
    返回:
        (原始图像, 姿势标签)
    """
    # 默认为未知姿势
    label = 'Unknown Pose'
    color = (0, 0, 255)  # 红色(BGR)
    
    # 如果没有提供角度，则计算角度
    if angles is None:
        angles = calculate_all_angles(landmarks)
    
    # 查找最匹配的姿势
    best_match, best_matches, best_score = find_best_matching_pose(angles)
    
    # 设置匹配阈值
    joint_weights = [1.0, 1.0, 1.5, 1.5, 1.3, 1.3]
    min_matches = 5  # 提高要求
    min_score = sum(joint_weights) * 0.7  # 要求至少70%的加权得分(原来是60%)
    
    # 判断是否匹配成功
    if best_match and best_matches >= min_matches and best_score >= min_score:
        label = best_match
        color = (0, 255, 0)  # 绿色(BGR)
        
        # 播放姿势指导语
        try:
            instruction = poses_data["poses"].get(label, {}).get("instruction", "")
            if not instruction:
                instruction = f"当前姿势是 {label}"
            Thread(target=speak_text, args=(instruction,), daemon=True).start()
        except Exception as e:
            print(f"读取指导语时出错: {e}")
            Thread(target=speak_text, args=(f"当前姿势是 {label}",), daemon=True).start()
        
        print(f"最佳匹配: {label}, 匹配关节数: {best_matches}/6, 得分: {best_score:.2f}/{sum(joint_weights):.2f}")
    
    # 不再在图像上绘制文本，直接返回原始图像和标签
    return output_image, label

# =========================================================
# 图像处理和姿势识别
# =========================================================
def process_pose_image(image_path):
    """
    处理图像并提取姿势角度数据
    
    参数:
        image_path: 图像文件路径
        
    返回:
        角度列表或None(如果检测失败)
    """
    # 读取图像
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"Error: 无法加载图像: {image_path}")
        return None
        
    # 调整图像大小保持纵横比
    height, width = image.shape[:2]
    target_height = 640
    scale = target_height / height
    target_width = int(width * scale)
    image = cv2.resize(image, (target_width, target_height))
    
    # 处理图像
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = static_pose.process(image_rgb)
    
    # 检查是否检测到姿势关键点
    if not results.pose_landmarks:
        print("Error: 未检测到姿势关键点。请确保图像中有清晰可见的人物。")
        return None
    
    # 转换关键点格式
    landmarks = []
    for landmark in results.pose_landmarks.landmark:
        landmarks.append((
            int(landmark.x * target_width),
            int(landmark.y * target_height),
            landmark.z * target_width
        ))
    
    try:
        # 计算关节角度
        angles = calculate_all_angles(landmarks)
        
        # 保存调试图像 - 使用绿色标记关键点
        debug_image = image.copy()
        
        # 创建绿色绘制规格
        draw_spec = mp_drawing.DrawingSpec(
            color=(0, 255, 0),
            thickness=2,
            circle_radius=2
        )
        
        # 绘制关键点和连接线
        mp_drawing.draw_landmarks(
            debug_image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=draw_spec,
            connection_drawing_spec=draw_spec
        )
        
        # 添加文本标签
        cv2.putText(debug_image, "已检测到姿势", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.8, (0, 255, 0), 2)  # 绿色文本
        
        # 保存调试图像
        debug_path = os.path.join(app.config['UPLOAD_FOLDER'], 'debug_landmarks.jpg')
        cv2.imwrite(debug_path, debug_image)
        
        return angles
        
    except Exception as e:
        print(f"计算角度时出错: {str(e)}")
        return None

def webcam_feed():
    """
    生成视频流帧，应用优化后的姿势检测和平滑处理
    
    生成器函数，用于Flask的视频流响应
    """
    global last_frame, latest_angles, landmarks_history
    
    # 初始化视频捕获
    if use_camera:
        print("使用摄像头源")
        camera_video = cv2.VideoCapture(0)  # 使用摄像头
        
        # 获取摄像头的实际分辨率
        actual_width = int(camera_video.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(camera_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"摄像头实际分辨率: {actual_width}x{actual_height}")
        
        # 根据实际分辨率计算最佳显示尺寸
        update_display_size(actual_width, actual_height)
        
        # 设置摄像头参数，但保持原始分辨率
        camera_video.set(cv2.CAP_PROP_FPS, 30)
        camera_video.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    else:
        print(f"使用视频文件: {current_video}")
        camera_video = cv2.VideoCapture(current_video)  # 使用视频文件
        
        # 获取视频的实际分辨率
        actual_width = int(camera_video.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(camera_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"视频实际分辨率: {actual_width}x{actual_height}")
        
        # 根据实际分辨率计算最佳显示尺寸
        update_display_size(actual_width, actual_height)
    
    # 检查摄像头/视频是否正确打开
    if not camera_video.isOpened():
        print(f"错误: 无法打开视频源: {'摄像头' if use_camera else current_video}")
        # 尝试更换为默认视频源
        try:
            default_path = video_directory + default_video
            print(f"尝试使用默认视频: {default_path}")
            camera_video = cv2.VideoCapture(default_path)
            if not camera_video.isOpened():
                print("错误: 默认视频也无法打开")
                return
        except Exception as e:
            print(f"尝试打开默认视频时发生错误: {e}")
            return
    
    # 设置视频捕获参数，提高稳定性
    try:
        # 设置缓冲区大小
        camera_video.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        # 设置帧率（如果可能）
        camera_video.set(cv2.CAP_PROP_FPS, 30)
    except Exception as e:
        print(f"设置视频参数时出错: {e}")
    
    # 定义绘制样式
    default_style = mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=3, circle_radius=2)  # 黄色
    success_style = mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=3, circle_radius=2)      # 绿色

    try:
        # 记录第一帧的尺寸，用于后续帧的标准化
        first_frame_size = None
        
        while camera_video.isOpened():
            # 读取视频帧
            ok, frame = camera_video.read()
            if not ok:
                # 视频结束时重新开始
                camera_video.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            
            # 记录第一帧的尺寸，用于后续帧的标准化
            if first_frame_size is None:
                first_frame_size = (frame.shape[1], frame.shape[0])  # (width, height)
                print(f"设置标准帧尺寸: {first_frame_size}")
            
            # 确保所有帧的尺寸一致，避免MediaPipe分割错误
            current_size = (frame.shape[1], frame.shape[0])
            if current_size != first_frame_size:
                # 如果尺寸不一致，调整到第一帧的尺寸
                print(f"检测到帧尺寸变化: {current_size} -> {first_frame_size}")
                frame = cv2.resize(frame, first_frame_size)
            
            # 动态调整帧大小到计算出的最佳显示尺寸
            display_width, display_height = get_display_size()
            frame = cv2.resize(frame, (display_width, display_height))
            
            # 添加状态文本 - 默认为未激活状态
            status_text = "状态: 未激活检测"
            status_color = (0, 0, 255)  # 红色(BGR)
            
            # 优化：减少锁操作，使用缓存的状态
            is_active, current_pose = get_recognition_state()
            
            # 如果已选择训练姿势但尚未激活检测，显示更有用的状态文本
            if current_pose and not is_active:
                status_text = f"准备训练: {current_pose}"
                status_color = (0, 165, 255)  # 橙色(BGR)
            
            if is_active:
                # 更新状态文本
                status_text = "状态: 检测中..."
                status_color = (0, 255, 0)  # 绿色(BGR)
                
                # 执行姿态检测和分类
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # 添加错误处理，避免MediaPipe分割错误导致程序崩溃
                try:
                    results = video_pose.process(frame_rgb)
                except Exception as e:
                    print(f"MediaPipe处理错误: {e}")
                    # 调用重置函数
                    reset_mediapipe_if_needed()
                    # 降级处理：跳过当前帧的姿态检测
                    results = None
                
                processed_frame = frame.copy()
                
                if results.pose_landmarks:
                    # 转换关键点
                    landmarks = []
                    for landmark in results.pose_landmarks.landmark:
                        landmarks.append((
                            int(landmark.x * frame.shape[1]),
                            int(landmark.y * frame.shape[0]),
                            landmark.z * frame.shape[1]
                        ))
                    
                    # 应用平滑处理
                    landmarks = smooth_landmark_data(landmarks)
                    
                    if landmarks:  # 确保平滑后的关键点不为空
                        try:
                            # 计算关节角度
                            angles = calculate_all_angles(landmarks)
                            
                            # 格式化角度数据
                            angle_data = {
                                'left_arm': round(angles[0], 2),
                                'right_arm': round(angles[1], 2),
                                'left_body': round(angles[2], 2),
                                'right_body': round(angles[3], 2),
                                'left_leg': round(angles[4], 2),
                                'right_leg': round(angles[5], 2)
                            }
                            
                            # 更新共享数据
                            try:
                                if pose_data_queue.full():
                                    pose_data_queue.get_nowait()
                                pose_data_queue.put_nowait(angle_data)
                            except queue.Full:
                                pass
                            
                            # 更新最新角度数据 - 优化锁操作
                            angle_data_dict = {
                                '左臂角度': round(angles[0], 2),
                                '右臂角度': round(angles[1], 2),
                                '左上身角度': round(angles[2], 2),
                                '右上身角度': round(angles[3], 2),
                                '左腿角度': round(angles[4], 2),
                                '右腿角度': round(angles[5], 2)
                            }
                            
                            # 批量更新，减少锁操作时间
                            with angles_lock:
                                latest_angles.update(angle_data_dict)

                            # 获取当前选择的训练姿势 - 使用已缓存的状态
                            # current_pose 已经在上面获取过了，不需要重复获取
                            
                            # 设置默认状态 - 黄色线条
                            pose_matched = False
                            status_text = "状态: 检测中..."
                            status_color = (0, 255, 255)  # 黄色(BGR)
                            current_style = default_style
                            label_info = None
                            
                            # 只有当选择了训练姿势时，才进行姿势匹配
                            if current_pose:
                                # 更新状态文本以显示当前正在训练的姿势
                                status_text = f"训练姿势: {current_pose}"
                                
                                # 仅匹配当前选择的姿势，不检查其他姿势
                                pose_matched = optimized_match_pose(current_pose, angles)
                                
                                if pose_matched:
                                    # 匹配成功 - 绿色线条
                                    current_style = success_style
                                    status_text = f"姿势正确: {current_pose}"
                                    status_color = (0, 255, 0)  # 绿色(BGR)
                                    label_info = (current_pose, (10, 60), (0, 255, 0))
                                    
                                    # 优化：减少频繁的语音播报
                                    instruction = poses_data["poses"].get(current_pose, {}).get("instruction", "")
                                    if not instruction:
                                        instruction = f"姿势正确"
                                    # 使用更高效的语音播报，避免频繁创建线程
                                    speak_text(instruction)
                                else:
                                    # 不匹配 - 黄色线条
                                    status_text = f"请调整到 {current_pose} 姿势"
                                    
                                    # 优化：减少频繁的语音播报
                                    speak_text("姿势不正确，请调整")
                            else:
                                # 未选择训练姿势时，不显示匹配信息
                                status_text = "请选择训练姿势"
                            
                            # 绘制关键点和连接线
                            mp_drawing.draw_landmarks(
                                processed_frame,
                                results.pose_landmarks,
                                mp_pose.POSE_CONNECTIONS,
                                landmark_drawing_spec=current_style,
                                connection_drawing_spec=current_style
                            )

                            # 移除姿势分类调用，我们只关注当前选择的训练姿势
                            # _, pose_label = classifyPose(landmarks, processed_frame.copy(), angles)
                            
                            # 准备后续在final_frame上添加姿势标签和状态文本
                            if pose_matched:
                                label_info = (current_pose, (10, 60), (0, 255, 0))
                            
                        except Exception as e:
                            print(f"姿态处理出错: {str(e)}")
                            # 出错时使用默认样式绘制
                            mp_drawing.draw_landmarks(
                                processed_frame,
                                results.pose_landmarks,
                                mp_pose.POSE_CONNECTIONS,
                                landmark_drawing_spec=default_style,
                                connection_drawing_spec=default_style
                            )
                            # 更新状态文本
                            status_text = "状态: 处理出错"
                            status_color = (0, 0, 255)  # 红色(BGR)
                            label_info = None
                else:
                    # 没有检测到姿势关键点
                    status_text = "未检测到姿势"
                    status_color = (0, 165, 255)  # 橙色(BGR)
                    label_info = None
                
                # 添加状态文本到处理后的图像 - 这里统一处理所有文本绘制
                img_pil = Image.fromarray(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB))
                draw = ImageDraw.Draw(img_pil)
                
                # 优化：使用缓存的字体，避免重复加载
                font = get_cached_font()
                
                # 绘制状态文本
                draw.text((10, 10), status_text, font=font, fill=status_color[::-1])
                
                # 如果有姿势标签，也一并绘制
                if label_info:
                    text, pos, color = label_info
                    draw.text(pos, text, font=font, fill=color)
                
                # 转换回OpenCV格式
                processed_frame = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
                
                # 更新最后处理的帧
                with frame_lock:
                    last_frame = processed_frame
            else:
                # 非活动状态下只显示原始视频帧并添加状态文本
                display_frame = frame.copy()
                # 移除黑色背景
                # 使用PIL添加中文状态文本
                img_pil = Image.fromarray(cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB))
                draw = ImageDraw.Draw(img_pil)
                
                # 优化：使用缓存的字体，避免重复加载
                font = get_cached_font()
                        
                # 在非活动状态下使用上面更新的status_text，它可能包含当前选择的姿势信息
                draw.text((10, 10), status_text, font=font, fill=status_color[::-1])
                display_frame = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
                with frame_lock:
                    last_frame = display_frame
                # 重置平滑数据
                landmarks_history = []

            # 获取当前要显示的帧
            with frame_lock:
                current_frame = last_frame.copy() if last_frame is not None else frame

            # 将帧转换为JPEG格式
            ret, jpeg = cv2.imencode('.jpg', current_frame)
            frame_data = jpeg.tobytes()

            # 返回帧数据
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n')
    finally:
        # 确保释放视频资源
        camera_video.release()

# =========================================================
# 辅助函数
# =========================================================
def allowed_file(filename):
    """
    检查文件名是否有效且为允许的类型
    """
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.template_filter('file_exists')
def file_exists(path):
    """
    模板过滤器：检查文件是否存在
    """
    return os.path.exists(path)

# =========================================================
# Flask路由
# =========================================================
@app.route('/')
def index():
    """首页路由"""
    return render_template('index.html')

@app.route('/yoga_try', endpoint='yoga_try_route')
def yoga_try():
    """瑜伽训练页面路由"""
    global use_camera, current_video
    
    # 获取视频源参数
    source = request.args.get('source', None)
    if source == 'camera':
        use_camera = True
    elif source == 'video':
        use_camera = False
        current_video = video_directory + default_video
    
    # 准备有效姿势数据
    pose_files = {}
    valid_poses = []
    
    # 确保上传目录存在
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    
    # 检查每个姿势的图片是否存在
    for pose in sum_data:
        # 获取文件名
        filename = poses_data["poses"].get(pose, {}).get("filename")
        
        # 如果未设置文件名，使用安全的文件名
        if not filename:
            safe_name = secure_filename(pose)
            if not safe_name:
                safe_name = f"pose_{sum_data.index(pose)}"
            filename = f"{safe_name}.jpg"
        
        # 检查文件是否存在
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if os.path.exists(file_path):
            pose_files[pose] = filename.replace('.jpg', '')  # 移除扩展名
            valid_poses.append(pose)
            print(f"找到姿势图片: '{pose}': {file_path}")
        else:
            print(f"警告: 未找到姿势图片 '{pose}': {file_path}")
    
    print("有效姿势:", valid_poses)
    
    # 获取视频目录中的所有视频文件
    available_videos = []
    if os.path.exists(video_directory):
        for file in os.listdir(video_directory):
            if file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                available_videos.append(file)
    
    # 传递当前视频源给模板
    return render_template('yoga_try.html', 
                          poses=valid_poses, 
                          pose_files=pose_files, 
                          is_camera=use_camera,
                          current_video=current_video,
                          available_videos=available_videos)

@app.route('/video_feed1')
def video_feed1():
    """视频流路由"""
    return Response(
        webcam_feed(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

@app.route('/angle_data')
def angle_data():
    """角度数据的SSE流"""
    def generate():
        while True:
            try:
                # 获取最新角度数据
                angle_data = pose_data_queue.get(timeout=1.0)
                yield f"data: {json.dumps(angle_data)}\n\n"
            except queue.Empty:
                # 队列为空时发送默认值
                default_data = {
                    'left_arm': 0, 'right_arm': 0,
                    'left_body': 0, 'right_body': 0,
                    'left_leg': 0, 'right_leg': 0
                }
                yield f"data: {json.dumps(default_data)}\n\n"
            time.sleep(1.0)  # 每秒更新一次

    return Response(generate(), mimetype='text/event-stream')

@app.route('/upload')
def upload():
    """上传姿势页面路由"""
    # 加载训练记录
    training_records = load_training_records()
    
    return render_template('upload.html', 
                          poses=sum_data, 
                          training_records=training_records.get("records", []))

@app.route('/upload_pose', methods=['POST'])
def upload_pose():
    """处理姿势上传"""
    # 检查是否有文件上传
    if 'pose_image' not in request.files:
        return '未上传文件', 400
    
    file = request.files['pose_image']
    pose_name = request.form['pose_name']
    instruction = request.form.get('pose_instruction', f"当前姿势是 {pose_name}")
    
    # 验证数据
    if pose_name in sum_data:
        return '该姿势名称已存在', 400
    
    if file.filename == '':
        return '未选择文件', 400
    
    if not file or not allowed_file(file.filename):
        return '不支持的文件类型', 400
    
    # 确保上传目录存在
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    
    # 使用时间戳创建唯一文件名
    timestamp = int(time.time())
    filename = f"pose_{timestamp}.jpg"
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    # 保存文件
    file.save(file_path)
    print(f"已保存文件: {file_path}")
    
    # 处理图像获取角度数据
    angles = process_pose_image(file_path)
    if angles is None:
        os.remove(file_path)
        return '图像中未检测到姿势', 400
    
    # 更新姿势数据
    poses_data["poses"][pose_name] = {
        "angles": angles,
        "instruction": instruction,
        "filename": filename
    }
    
    # 保存更新后的数据
    with open('poses_data.json', 'w', encoding='utf-8') as f:
        json.dump(poses_data, f, ensure_ascii=False, indent=2)
    
    # 更新姿势列表
    sum_data.append(pose_name)
    
    print(f"成功保存姿势 {pose_name}, 角度: {angles}")
    return redirect(url_for('yoga_try_route'))

@app.route('/delete_pose/<pose_name>')
def delete_pose(pose_name):
    """删除姿势"""
    try:
        # 获取姿势对应的文件名
        filename = poses_data["poses"].get(pose_name, {}).get("filename")
        
        # 删除文件（如果存在）
        if filename:
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            if os.path.exists(image_path):
                os.remove(image_path)
                print(f"已删除图像文件: {image_path}")
        
        # 从数据中删除姿势
        if pose_name in poses_data["poses"]:
            del poses_data["poses"][pose_name]
            
        # 保存更新后的数据
        with open('poses_data.json', 'w', encoding='utf-8') as f:
            json.dump(poses_data, f, ensure_ascii=False, indent=2)
        
        # 更新姿势列表
        if pose_name in sum_data:
            sum_data.remove(pose_name)
        
        return redirect(url_for('upload'))
    except Exception as e:
        print(f"删除姿势时出错: {str(e)}")
        return f"删除姿势时出错: {str(e)}", 500

@app.route('/get_angles')
def get_angles():
    """获取当前角度数据"""
    with angles_lock:
        return json.dumps(latest_angles)

@app.route('/toggle_recognition', methods=['POST'])
def toggle_recognition():
    """切换识别状态"""
    global recognition_active
    data = request.json
    active = data.get('active', False)
    
    # 更新全局识别状态
    with recognition_lock:
        recognition_active = active
    
    return json.dumps({'active': active})

@app.route('/start_training', methods=['POST'])
def start_training():
    """开始瑜伽训练"""
    global recognition_active, current_training_pose
    
    try:
        data = request.json
        pose_name = data.get('pose_name', '')
        
        # 检查姿势是否存在
        if not pose_name or pose_name not in poses_data.get("poses", {}):
            return json.dumps({
                'success': False,
                'message': 'Pose not found'
            })
        
        # 激活姿势识别并设置当前训练姿势
        with recognition_lock:
            recognition_active = True
            current_training_pose = pose_name
            print(f"当前训练姿势设置为: {current_training_pose}")
        
        # 重置缓存状态，确保新的训练状态能正确更新
        reset_caches()
        
        return json.dumps({
            'success': True,
            'message': 'Training started'
        })
            
    except Exception as e:
        print(f"Start training error: {str(e)}")
        return json.dumps({
            'success': False,
            'message': f'Error: {str(e)}'
        })

@app.route('/end_training', methods=['POST'])
def end_training():
    """结束瑜伽训练"""
    global recognition_active, current_training_pose
    
    try:
        # 停止姿势识别并重置当前训练姿势
        with recognition_lock:
            recognition_active = False
            current_training_pose = None
            print("训练结束，重置训练姿势")
        
        # 重置缓存状态
        reset_caches()
        
        return json.dumps({
            'success': True,
            'message': 'Training ended'
        })
            
    except Exception as e:
        print(f"End training error: {str(e)}")
        return json.dumps({
            'success': False,
            'message': f'Error: {str(e)}'
        })

@app.route('/get_standard_pose_angles', methods=['POST'])
def get_standard_pose_angles():
    """获取标准姿势角度数据"""
    data = request.json
    pose_name = data.get('pose_name', '')
    
    # 检查姿势是否存在
    if not pose_name or pose_name not in poses_data.get("poses", {}):
        return json.dumps({
            'error': 'Pose not found',
            'leftArm': 0, 'rightArm': 0,
            'leftBody': 0, 'rightBody': 0,
            'leftLeg': 0, 'rightLeg': 0
        })
    
    # 获取标准角度数据
    angles = poses_data["poses"][pose_name].get("angles", [0, 0, 0, 0, 0, 0])
    
    # 返回格式化数据
    return json.dumps({
        'leftArm': angles[0],
        'rightArm': angles[1],
        'leftBody': angles[2],
        'rightBody': angles[3],
        'leftLeg': angles[4],
        'rightLeg': angles[5]
    })

# =========================================================
# 数据平滑处理
# =========================================================
def smooth_landmark_data(current_landmarks):
    """
    平滑关键点数据，减少抖动和噪声
    
    参数:
        current_landmarks: 当前检测到的关键点
        
    返回:
        平滑处理后的关键点
    """
    global landmarks_history
    
    # 如果当前关键点为空，返回None
    if not current_landmarks:
        return None
    
    # 添加当前关键点到历史记录
    landmarks_history.append(current_landmarks)
    
    # 保持历史记录在合理长度
    if len(landmarks_history) > MAX_HISTORY_LENGTH:
        landmarks_history.pop(0)
    
    # 如果历史记录太少，直接返回当前关键点
    if len(landmarks_history) < 2:
        return current_landmarks
    
    # 应用加权平均平滑
    smoothed_landmarks = []
    for i in range(len(current_landmarks)):
        # 提取历史记录中当前关键点的x, y, z坐标
        x_vals = [lm[i][0] for lm in landmarks_history]
        y_vals = [lm[i][1] for lm in landmarks_history]
        z_vals = [lm[i][2] for lm in landmarks_history]
        
        # 更近的帧权重更高
        weights = [0.1, 0.2, 0.3, 0.4, 0.5][-len(landmarks_history):]
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]
        
        # 计算加权平均
        smooth_x = sum(x * w for x, w in zip(x_vals, weights))
        smooth_y = sum(y * w for y, w in zip(y_vals, weights))
        smooth_z = sum(z * w for z, w in zip(z_vals, weights))
        
        smoothed_landmarks.append((smooth_x, smooth_y, smooth_z))
    
    return smoothed_landmarks

# =========================================================
# 训练记录相关代码
# =========================================================
# 训练记录文件
TRAINING_RECORDS_FILE = 'training_records.json'

def load_training_records():
    """加载训练记录，如果不存在则创建新文件"""
    try:
        with open(TRAINING_RECORDS_FILE, 'r', encoding='utf-8') as file:
            return json.load(file)
    except FileNotFoundError:
        # 如果文件不存在，创建新的记录结构
        records = {"records": []}
        with open(TRAINING_RECORDS_FILE, 'w', encoding='utf-8') as file:
            json.dump(records, file, ensure_ascii=False, indent=2)
        return records

def save_training_records(records):
    """保存训练记录到文件"""
    with open(TRAINING_RECORDS_FILE, 'w', encoding='utf-8') as file:
        json.dump(records, ensure_ascii=False, indent=2, fp=file)

@app.route('/save_training_record', methods=['POST'])
def save_training_record():
    """保存训练记录"""
    try:
        data = request.json
        
        # 检查必需字段
        if not all(key in data for key in ['pose_name', 'user_angles', 'standard_angles', 'match_count', 'match_percentage']):
            return json.dumps({
                'success': False,
                'message': 'Missing required fields'
            })
        
        # 读取当前记录
        records = load_training_records()
        
        # 生成记录ID
        record_id = str(uuid.uuid4())
        
        # 计算每个关节的详细数据
        angle_details = {}
        joint_names = ['leftArm', 'rightArm', 'leftBody', 'rightBody', 'leftLeg', 'rightLeg']
        
        # 定义每个关节的误差容忍度
        max_errors = {
            'leftArm': 25, 'rightArm': 25,
            'leftBody': 20, 'rightBody': 20,
            'leftLeg': 25, 'rightLeg': 25
        }
        
        non_matching_joints = []
        improvement_suggestions = {}
        
        for joint in joint_names:
            user_angle = data['user_angles'].get(joint, 0)
            std_angle = data['standard_angles'].get(joint, 0)
            
            # 计算差异和匹配状态
            difference = abs(user_angle - std_angle)
            is_matching = difference <= max_errors.get(joint, 30)
            
            # 计算偏差百分比
            deviation_percent = (difference / std_angle * 100) if std_angle != 0 else 0
            
            # 确定改进方向
            direction = "过大" if user_angle > std_angle else "过小"
            
            # 记录不匹配的关节
            if not is_matching:
                non_matching_joints.append(joint)
                
                # 添加具体建议
                if joint.endswith('Arm'):
                    if direction == "过大":
                        suggestion = f"尝试减小{joint}角度，放松肘部"
                    else:
                        suggestion = f"尝试增大{joint}角度，伸直手臂"
                elif joint.endswith('Body'):
                    if direction == "过大":
                        suggestion = f"尝试减小{joint}角度，收紧躯干"
                    else:
                        suggestion = f"尝试增大{joint}角度，延展上身"
                elif joint.endswith('Leg'):
                    if direction == "过大":
                        suggestion = f"尝试减小{joint}角度，弯曲膝盖"
                    else:
                        suggestion = f"尝试增大{joint}角度，伸直腿部"
                
                improvement_suggestions[joint] = suggestion
            
            # 保存详细数据
            angle_details[joint] = {
                "user": user_angle,
                "standard": std_angle,
                "difference": difference,
                "deviation_percent": round(deviation_percent, 2),
                "is_matching": is_matching,
                "direction": direction if not is_matching else "正常"
            }
        
        # 与历史记录比较
        historical_comparison = None
        previous_records = [r for r in records.get('records', []) 
                          if r.get('pose_name') == data['pose_name']]
        
        if previous_records:
            last_record = previous_records[-1]
            improved_joints = []
            regressed_joints = []
            
            for joint in joint_names:
                prev_deviation = abs(last_record.get('user_angles', {}).get(joint, 0) - 
                                    last_record.get('standard_angles', {}).get(joint, 0))
                curr_deviation = abs(data['user_angles'].get(joint, 0) - 
                                    data['standard_angles'].get(joint, 0))
                
                if prev_deviation > curr_deviation:
                    improved_joints.append(joint)
                elif prev_deviation < curr_deviation:
                    regressed_joints.append(joint)
            
            # 计算整体进步程度
            if len(improved_joints) > len(regressed_joints):
                progress = "进步"
            elif len(improved_joints) < len(regressed_joints):
                progress = "退步"
            else:
                progress = "持平"
                
            historical_comparison = {
                "previous_date": last_record.get('timestamp', '').split('T')[0],
                "improved_joints": improved_joints,
                "regressed_joints": regressed_joints,
                "overall_progress": progress
            }
        
        # 获取呼吸指导(如果可用)
        yoga_knowledge = {}
        pose_description = ""
        
        # 尝试使用AI生成瑜伽呼吸指导
        if DEEPSEEK_API_KEY and non_matching_joints and data.get('use_ai_suggestions', True):
            try:
                pose_name = data['pose_name']
                yoga_knowledge = generate_yoga_knowledge(pose_name)
                
                # 提取姿势描述
                if yoga_knowledge.get('description'):
                    pose_description = yoga_knowledge.get('description')
                    
            except Exception as e:
                print(f"获取AI瑜伽知识时出错: {str(e)}")
                # 出错时使用默认建议，不影响整体功能
        
        # 创建增强的记录
        record = {
            'id': record_id,
            'pose_name': data['pose_name'],
            'timestamp': datetime.now().isoformat(),
            'user_angles': data['user_angles'],
            'standard_angles': data['standard_angles'],
            'match_count': data['match_count'],
            'match_percentage': data['match_percentage'],
            'used_swapped_legs': data.get('used_swapped_legs', False),
            'angle_details': angle_details,
            'non_matching_joints': non_matching_joints,
            'improvement_suggestions': improvement_suggestions,
            'historical_comparison': historical_comparison
        }
        
        # 添加AI生成的呼吸指导(如果有)
        if pose_description:
            record['pose_description'] = pose_description
        if yoga_knowledge:
            record['yoga_knowledge'] = {
                'breathing': yoga_knowledge.get('breathing', '')
            }
        
        # 添加记录
        records['records'].append(record)
        
        # 保存到文件
        with open('training_records.json', 'w', encoding='utf-8') as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
        
        return json.dumps({
            'success': True,
            'message': 'Record saved successfully'
        })
        
    except Exception as e:
        print(f"保存训练记录出错: {str(e)}")
        traceback.print_exc()
        return json.dumps({
            'success': False,
            'message': f'Error: {str(e)}'
        })

@app.route('/get_training_records')
def get_training_records():
    """获取所有训练记录"""
    try:
        training_records = load_training_records()
        return json.dumps(training_records)
    except Exception as e:
        print(f"获取训练记录时出错: {str(e)}")
        return json.dumps({"success": False, "message": f"获取失败: {str(e)}"}), 500

@app.route('/delete_training_record/<string:record_id>', methods=['GET'])
def delete_training_record(record_id):
    """删除指定ID的训练记录"""
    try:
        # 加载现有记录
        training_records = load_training_records()
        
        # 查找并删除匹配ID的记录
        records = training_records.get("records", [])
        for i, record in enumerate(records):
            if record.get("id") == record_id:
                records.pop(i)
                break
        
        # 保存更新后的记录
        save_training_records(training_records)
        
        # 重定向回上传页面
        return redirect(url_for('upload'))
        
    except Exception as e:
        print(f"删除训练记录时出错: {str(e)}")
        return json.dumps({"success": False, "message": f"删除失败: {str(e)}"}), 500

@app.route('/yoga_ai_chat')
def yoga_ai_chat():
    """AI对话页面路由"""
    # 加载训练记录用于显示
    training_records = load_training_records()
    record_count = len(training_records.get("records", []))
    
    return render_template('yoga_ai_chat.html', 
                           record_count=record_count,
                           has_records=(record_count > 0))

# =========================================================
# Qwen AI模型集成 - 重构优化版本
# =========================================================
# SiliconFlow API配置 (支持Qwen模型)
DEEPSEEK_API_URL = "https://api.siliconflow.cn/v1/chat/completions"
DEEPSEEK_API_KEY = "sk-ssewlhnmamzhqgsumgdrraguhkynrbrcxnxyeaoxqordwmyj"

# API配置常量
API_TIMEOUT = 20  # 减少超时时间以快速失败
MAX_RETRIES = 1   # 减少重试次数以提升速度  
RETRY_DELAY = 0.5 # 减少重试间隔

# AI模型配置
class ModelConfig:
    """AI模型配置类"""
    def __init__(self):
        self.model = "Qwen/Qwen3-30B-A3B"
        self.temperature = 0.6  # 降低温度以加快响应
        self.max_tokens = 800   # 减少token数量以提升速度
        self.top_p = 0.85      # 降低top_p以加快采样
        self.frequency_penalty = 0.1  # 降低频率惩罚
        self.presence_penalty = 0.1   # 降低存在惩罚
        self.stream = False
    
    def to_dict(self):
        """转换为字典格式"""
        return {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty,
            "stream": self.stream
        }

# 提示词配置
class PromptConfig:
    """提示词配置类"""
    def __init__(self):
        self.max_records = None  # 移除历史记录数量限制
        self.max_poses = None    # 移除姿势数量限制
        self.question_types = {
            "analysis": ["分析", "评估", "情况", "进展"],
            "suggestion": ["建议", "改进", "如何", "怎么"],
            "plan": ["计划", "安排", "训练"]
        }
    
    def get_question_type(self, user_query: str) -> str:
        """判断问题类型"""
        query_lower = user_query.lower()
        for qtype, keywords in self.question_types.items():
            if any(keyword in query_lower for keyword in keywords):
                return qtype
        return "default"

# 训练记录数据类
class TrainingRecord:
    """训练记录数据类"""
    def __init__(self, data: dict):
        self.pose_name = data.get("pose_name", "")
        self.match_percentage = data.get("match_percentage", 0)
        self.timestamp = data.get("timestamp", "")
        self.non_matching_joints = data.get("non_matching_joints", [])
        self.angle_details = data.get("angle_details", {})
        self.improvement_suggestions = data.get("improvement_suggestions", {})
    
    def get_date(self) -> str:
        """获取训练日期"""
        return self.timestamp.split("T")[0] if self.timestamp else "未知时间"
    
    def format_for_prompt(self) -> str:
        """格式化训练记录用于提示词"""
        lines = [
            f"{self.pose_name} (训练时间: {self.get_date()})",
            f"   匹配度: {self.match_percentage}%"
        ]
        
        if self.non_matching_joints:
            lines.append(f"   需改进关节: {', '.join(self.non_matching_joints)}")
        
        if self.angle_details:
            lines.extend(self._format_angle_details())
        
        if self.improvement_suggestions:
            lines.append(f"   改进建议: {', '.join(self.improvement_suggestions.values())}")
        
        return "\n".join(lines)
    
    def _format_angle_details(self) -> list:
        """格式化角度详情 - 无限制版本"""
        lines = []
        # 显示所有不匹配的关节
        for joint, details in self.angle_details.items():
            if not details.get("is_matching", True):
                user_angle = details.get("user", 0)
                std_angle = details.get("standard", 0)
                deviation = details.get("deviation_percent", 0)
                lines.append(f"     - {joint}: 用户{user_angle}° vs 标准{std_angle}° (偏差{deviation}%)")
        return lines if lines else []

# 提示词生成器
class PromptGenerator:
    """提示词生成器"""
    def __init__(self):
        self.config = PromptConfig()
        self.instructions = {
            "analysis": """请详细分析用户的训练情况，包括：
- 训练进展和趋势分析
- 存在的问题和原因
- 各关节的具体表现
- 整体改进方向""",
            
            "suggestion": """请提供具体的、可操作的改进建议，包括：
- 针对性的动作要领
- 详细的调整方法
- 注意事项和禁忌
- 练习频率建议""",
            
            "plan": """请制定个性化的训练计划，包括：
- 基于当前水平的计划安排
- 循序渐进的目标设定
- 具体的练习内容
- 时间安排建议""",
            
            "default": """请根据用户的具体问题提供专业、详细、个性化的回答，包括：
- 针对性的分析和建议
- 实用的改进方法
- 鼓励和支持的话语"""
        }
    
    def generate(self, user_query: str, training_records: list, poses_data: dict) -> str:
        """生成完整的提示词 - 优化版本"""
        # 基础提示词 - 简化版本
        prompt = "你是瑜伽教练AI助手。基于训练记录提供指导。\n\n训练记录："
        
        # 添加训练记录 - 无限制
        if training_records:
            for i, record in enumerate(training_records):
                prompt += f"\n{i+1}. {record.format_for_prompt()}"
        else:
            prompt += "\n暂无记录"
        
        # 添加可用姿势 - 无限制
        available_poses = list(poses_data.get("poses", {}).keys())
        if available_poses:
            prompt += f"\n\n可用姿势: {', '.join(available_poses)}"
        
        # 添加问题类型特定的指令 - 简化版本
        question_type = self.config.get_question_type(user_query)
        prompt += f"\n\n{self._get_simplified_instruction(question_type)}"
        
        return prompt
    
    def _get_simplified_instruction(self, question_type: str) -> str:
        """获取简化的指令"""
        simplified_instructions = {
            "analysis": "请简要分析训练情况，指出主要问题和改进方向。",
            "suggestion": "请提供具体的改进建议和练习方法。",
            "plan": "请制定简单的训练计划和时间安排。",
            "default": "请根据问题提供专业、简洁的建议。"
        }
        return simplified_instructions.get(question_type, simplified_instructions["default"])

def make_api_request_with_retry(url, payload, headers, max_retries=MAX_RETRIES):
    """
    带重试机制的API请求函数
    
    参数:
        url: API端点URL
        payload: 请求数据
        headers: 请求头
        max_retries: 最大重试次数
        
    返回:
        API响应数据或None
    """
    import time
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
    
    # 创建会话并配置重试策略
    session = requests.Session()
    
    # 配置重试策略
    retry_strategy = Retry(
        total=max_retries,
        backoff_factor=1,  # 指数退避
        status_forcelist=[429, 500, 502, 503, 504, 520, 521, 522, 523, 524],
        allowed_methods=["POST", "GET"],
        respect_retry_after_header=True
    )
    
    # 配置适配器
    adapter = HTTPAdapter(max_retries=retry_strategy, pool_connections=10, pool_maxsize=20)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    
    # 设置连接池和超时
    session.headers.update(headers)
    
    try:
        print(f"正在发送API请求到: {url}")
        print(f"请求模型: {payload.get('model', 'unknown')}")
        print(f"消息数量: {len(payload.get('messages', []))}")
        
        # 发送请求 - 优化超时设置
        response = session.post(
            url, 
            json=payload, 
            timeout=(3, API_TIMEOUT),  # (连接超时3秒, 读取超时20秒)
            verify=True
        )
        
        # 检查响应状态
        response.raise_for_status()
        
        # 解析响应
        result = response.json()
        print(f"API请求成功，响应状态码: {response.status_code}")
        
        return result
        
    except requests.exceptions.Timeout as e:
        print(f"API请求超时: {str(e)}")
        return None
    except requests.exceptions.ConnectionError as e:
        print(f"连接错误: {str(e)}")
        return None
    except requests.exceptions.HTTPError as e:
        status_code = e.response.status_code if hasattr(e, 'response') else 'unknown'
        print(f"HTTP错误 {status_code}: {str(e)}")
        
        # 尝试解析错误响应
        try:
            error_data = e.response.json()
            error_code = error_data.get('code', 'unknown')
            error_message = error_data.get('message', str(e))
            print(f"错误代码: {error_code}, 错误信息: {error_message}")
        except:
            pass
            
        return None
    except Exception as e:
        print(f"API请求发生未知错误: {str(e)}")
        return None
    finally:
        session.close()

# API客户端类
class APIClient:
    """API客户端类"""
    def __init__(self, api_key: str, api_url: str):
        self.api_key = api_key
        self.api_url = api_url
    
    def call(self, model_config: ModelConfig, system_prompt: str, user_query: str) -> dict:
        """调用API"""
        payload = {
            **model_config.to_dict(),
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_query}
            ]
        }
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "User-Agent": "YogaAI/1.0"
        }
        
        return make_api_request_with_retry(self.api_url, payload, headers)

# 响应解析器类
class ResponseParser:
    """响应解析器类"""
    def parse(self, result: dict) -> str:
        """解析API响应"""
        if result is None:
            return "抱歉，AI服务暂时不可用，请稍后再试。"
        
        try:
            choices = result.get("choices", [])
            if not choices:
                print("API响应中没有choices字段")
                return "抱歉，AI响应格式异常，请稍后再试。"
            
            message = choices[0].get("message", {})
            generated_text = message.get("content", "")
            
            if not generated_text:
                print("API响应中没有生成文本")
                return "抱歉，AI未能生成有效回复，请稍后再试。"
            
            print(f"成功生成回复，长度: {len(generated_text)} 字符")
            return generated_text
            
        except Exception as e:
            print(f"解析API响应时出错: {str(e)}")
            print(f"原始响应: {result}")
            return "抱歉，处理AI响应时出错，请稍后再试。"

# 错误处理器类
class ErrorHandler:
    """错误处理器类"""
    @staticmethod
    def handle_error(error: Exception) -> str:
        """处理错误"""
        print(f"生成Qwen响应时出错: {str(error)}")
        import traceback
        traceback.print_exc()
        return "抱歉，处理您的请求时发生错误，请稍后再试。"

# 主AI助手类
class YogaAIAssistant:
    """瑜伽AI助手主类"""
    def __init__(self):
        self.model_config = ModelConfig()
        self.prompt_generator = PromptGenerator()
        self.api_client = APIClient(DEEPSEEK_API_KEY, DEEPSEEK_API_URL)
        self.response_parser = ResponseParser()
        self.error_handler = ErrorHandler()
        self.fast_mode = True  # 启用快速模式
    
    def generate_response(self, user_query: str, training_records: dict, poses_data: dict) -> str:
        """生成AI响应 - 优化版本"""
        # 检查API密钥
        if not DEEPSEEK_API_KEY:
            print("API密钥未配置")
            return "API暂时不可用，请确保已配置有效的API密钥。"
        
        try:
            # 快速模式优化
            if self.fast_mode:
                # 1. 准备训练记录数据（限制数量）
                recent_records = self._prepare_training_records(training_records)
                
                # 2. 生成简化提示词
                system_prompt = self.prompt_generator.generate(user_query, recent_records, poses_data)
                
                # 3. 使用快速模型配置
                fast_config = self._get_fast_config()
                result = self.api_client.call(fast_config, system_prompt, user_query)
                
                # 4. 解析响应
                return self.response_parser.parse(result)
            else:
                # 标准模式
                recent_records = self._prepare_training_records(training_records)
                system_prompt = self.prompt_generator.generate(user_query, recent_records, poses_data)
                result = self.api_client.call(self.model_config, system_prompt, user_query)
                return self.response_parser.parse(result)
            
        except Exception as e:
            return self.error_handler.handle_error(e)
    
    def _get_fast_config(self) -> ModelConfig:
        """获取快速模式配置"""
        fast_config = ModelConfig()
        fast_config.temperature = 0.5  # 更低温度
        fast_config.max_tokens = 500   # 更少token
        fast_config.top_p = 0.8       # 更低top_p
        fast_config.frequency_penalty = 0.0  # 无频率惩罚
        fast_config.presence_penalty = 0.0   # 无存在惩罚
        return fast_config
    
    def _prepare_training_records(self, training_records: dict) -> list:
        """准备训练记录数据 - 无限制版本"""
        records = training_records.get("records", [])
        # 移除数量限制，显示所有记录
        
        return [TrainingRecord(record) for record in records]

# 创建全局AI助手实例
ai_assistant = YogaAIAssistant()

def generate_qwen_response(user_query, training_records, poses_data):
    """
    调用Qwen API生成对话响应 - 重构版本
    
    参数:
        user_query: 用户查询文本
        training_records: 训练记录数据
        poses_data: 姿势数据
    
    返回:
        AI生成的回复文本
    """
    return ai_assistant.generate_response(user_query, training_records, poses_data)

# 瑜伽知识生成器类
class YogaKnowledgeGenerator:
    """瑜伽知识生成器类"""
    def __init__(self):
        self.api_client = APIClient(DEEPSEEK_API_KEY, DEEPSEEK_API_URL)
        self.model_config = ModelConfig()
        self.model_config.temperature = 0.6
        self.model_config.max_tokens = 1200
        # 确保使用新的模型
        self.model_config.model = "Qwen/Qwen3-30B-A3B"
    
    def generate(self, pose_name: str) -> dict:
        """生成瑜伽姿势知识"""
        if not DEEPSEEK_API_KEY:
            print("API密钥未配置，无法生成瑜伽知识")
            return {}
        
        try:
            # 生成提示词
            system_prompt = self._create_knowledge_prompt(pose_name)
            
            # 调用API
            result = self.api_client.call(self.model_config, system_prompt, f"请为{pose_name}生成专业的瑜伽知识")
            
            # 解析响应
            return self._parse_knowledge_response(result, pose_name)
            
        except Exception as e:
            print(f"生成瑜伽知识时出错: {str(e)}")
            import traceback
            traceback.print_exc()
            return {}
    
    def _create_knowledge_prompt(self, pose_name: str) -> str:
        """创建知识生成提示词"""
        return f"""你是一位专业的瑜伽教练。请为"{pose_name}"瑜伽姿势提供简洁的呼吸指导，以JSON格式回复：

{{
  "pose_name": "{pose_name}",
  "description": "2-3句话描述正确形态",
  "breathing": "针对这个姿势的具体呼吸指导，包括呼吸节奏、注意事项等"
}}

确保JSON格式正确，内容简洁实用。"""
    
    def _parse_knowledge_response(self, result: dict, pose_name: str) -> dict:
        """解析知识生成响应"""
        if result is None:
            print(f"无法为'{pose_name}'生成瑜伽知识")
            return {}
        
        try:
            choices = result.get("choices", [])
            if not choices:
                return {}
            
            message = choices[0].get("message", {})
            generated_text = message.get("content", "")
            
            if not generated_text:
                print(f"未能生成'{pose_name}'的瑜伽知识")
                return {}
            
            # 提取JSON部分
            json_text = self._extract_json_from_text(generated_text)
            
            # 解析JSON
            yoga_knowledge = json.loads(json_text)
            print(f"成功生成'{pose_name}'的瑜伽知识")
            return yoga_knowledge
            
        except json.JSONDecodeError as e:
            print(f"解析AI生成的瑜伽知识JSON失败: {e}")
            print(f"原始文本: {generated_text}")
            return {}
        except Exception as e:
            print(f"解析瑜伽知识API响应时出错: {str(e)}")
            return {}
    
    def _extract_json_from_text(self, text: str) -> str:
        """从文本中提取JSON"""
        if "```json" in text:
            return text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            return text.split("```")[1].split("```")[0].strip()
        return text

# 创建全局瑜伽知识生成器实例
yoga_knowledge_generator = YogaKnowledgeGenerator()

def generate_yoga_knowledge(pose_name):
    """
    利用AI生成瑜伽姿势的专业知识库数据 - 重构版本
    
    参数:
        pose_name: 瑜伽姿势名称
        
    返回:
        包含姿势专业知识的字典
    """
    return yoga_knowledge_generator.generate(pose_name)

@app.route('/ai_chat_query', methods=['POST'])
def ai_chat_query():
    """处理AI对话请求 - 优化版本"""
    try:
        query_data = request.json
        user_query = query_data.get("query", "")
        
        if not user_query:
            return json.dumps({"success": False, "message": "请提供查询内容"}), 400
        
        print(f"收到AI对话请求: {user_query[:50]}...")
        
        # 加载训练记录和姿势数据
        training_records = load_training_records()
        
        # 调用Qwen API生成响应
        ai_response = generate_qwen_response(user_query, training_records, poses_data)
        
        response_data = {
            "success": True,
            "response": ai_response
        }
        
        print("AI对话请求处理完成")
        return json.dumps(response_data, ensure_ascii=False)
        
    except Exception as e:
        print(f"处理AI对话请求时出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return json.dumps({"success": False, "message": f"请求失败: {str(e)}"}), 500

# 添加视频文件上传和选择路由
@app.route('/upload_video_file', methods=['POST'])
def upload_video_file():
    """处理视频文件上传"""
    global current_video, use_camera
    
    if 'video_file' not in request.files:
        return redirect(url_for('yoga_try_route'))
    
    file = request.files['video_file']
    if file.filename == '':
        return redirect(url_for('yoga_try_route'))
    
    # 确保视频目录存在
    if not os.path.exists(video_directory):
        os.makedirs(video_directory)
    
    # 安全地保存文件
    filename = secure_filename(file.filename)
    file_path = os.path.join(video_directory, filename)
    file.save(file_path)
    
    # 更新当前视频
    current_video = file_path
    use_camera = False
    
    print(f"已上传视频文件: {current_video}")
    
    return redirect(url_for('yoga_try_route'))

@app.route('/select_video', methods=['POST'])
def select_video():
    """选择现有视频文件"""
    global current_video, use_camera
    
    video_name = request.form.get('video_name')
    if video_name and os.path.exists(os.path.join(video_directory, video_name)):
        current_video = os.path.join(video_directory, video_name)
        use_camera = False
        print(f"已选择视频: {current_video}")
    else:
        print(f"选择的视频不存在: {video_name}")
    
    return redirect(url_for('yoga_try_route'))

@app.route('/get_display_size')
def get_display_size_api():
    """获取当前显示尺寸的API端点"""
    display_width, display_height = get_display_size()
    return json.dumps({
        'width': display_width,
        'height': display_height,
        'aspect_ratio': display_width / display_height if display_height > 0 else 1.0
    })

# =========================================================
# 应用入口
# =========================================================
if __name__ == '__main__':
    app.run(debug=True)

@app.errorhandler(Exception)
def handle_exception(e):
    print(f"应用错误: {e}")
    return "应用发生错误，请检查控制台日志", 500

# 在应用退出前释放资源
@app.teardown_appcontext
def release_resources(exception=None):
    try:
        static_pose.close()
        video_pose.close()
        print("MediaPipe资源已释放")
    except Exception as e:
        print(f"释放资源时出错: {e}")
