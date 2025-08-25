import os
import shutil
import random

def move_images_and_annotations(image_src, image_dest, annotation_src, annotation_dest, move_count):
    """
    随机将指定数量的图片和对应的注释文件从 validation 移动到 training。
    
    :param image_src: 图片源文件夹路径 (e.g., "images/validation")
    :param image_dest: 图片目标文件夹路径 (e.g., "images/training")
    :param annotation_src: 注释源文件夹路径 (e.g., "annotations/validation")
    :param annotation_dest: 注释目标文件夹路径 (e.g., "annotations/training")
    :param move_count: 要移动的图片数量
    """
    # 获取源文件夹中的所有图片文件
    image_files = [f for f in os.listdir(image_src) if os.path.isfile(os.path.join(image_src, f))]
    
    # 确保移动数量不超过文件总数
    move_count = min(move_count, len(image_files))
    
    # 随机选择要移动的文件
    files_to_move = random.sample(image_files, move_count)
    
    for file_name in files_to_move:
        # 源图片和目标图片路径
        image_path_src = os.path.join(image_src, file_name)
        image_path_dest = os.path.join(image_dest, file_name)

        # 移动图片
        if os.path.exists(image_path_src):
            shutil.move(image_path_src, image_path_dest)
            print(f"Moved image: {file_name} -> {image_dest}")
        
        # 对应的注释文件名
        annotation_name = os.path.splitext(file_name)[0] + ".0.png"  
        annotation_path_src = os.path.join(annotation_src, annotation_name)
        annotation_path_dest = os.path.join(annotation_dest, annotation_name)
        
        # 移动注释文件
        if os.path.exists(annotation_path_src):
            shutil.move(annotation_path_src, annotation_path_dest)
            print(f"Moved annotation: {annotation_name} -> {annotation_dest}")

        # 对应的注释文件名
        annotation_name = os.path.splitext(file_name)[0] + ".1.png"  
        annotation_path_src = os.path.join(annotation_src, annotation_name)
        annotation_path_dest = os.path.join(annotation_dest, annotation_name)
        
        # 移动注释文件
        if os.path.exists(annotation_path_src):
            shutil.move(annotation_path_src, annotation_path_dest)
            print(f"Moved annotation: {annotation_name} -> {annotation_dest}")
        
        # 对应的注释文件名
        annotation_name = os.path.splitext(file_name)[0] + ".2.png"  
        annotation_path_src = os.path.join(annotation_src, annotation_name)
        annotation_path_dest = os.path.join(annotation_dest, annotation_name)
        
        # 移动注释文件
        if os.path.exists(annotation_path_src):
            shutil.move(annotation_path_src, annotation_path_dest)
            print(f"Moved annotation: {annotation_name} -> {annotation_dest}")
        
        # 对应的注释文件名
        annotation_name = os.path.splitext(file_name)[0] + ".3.png"  
        annotation_path_src = os.path.join(annotation_src, annotation_name)
        annotation_path_dest = os.path.join(annotation_dest, annotation_name)
        
        # 移动注释文件
        if os.path.exists(annotation_path_src):
            shutil.move(annotation_path_src, annotation_path_dest)
            print(f"Moved annotation: {annotation_name} -> {annotation_dest}")
        
        # 对应的注释文件名
        annotation_name = os.path.splitext(file_name)[0] + ".3db.png"  
        annotation_path_src = os.path.join(annotation_src, annotation_name)
        annotation_path_dest = os.path.join(annotation_dest, annotation_name)
        
        # 移动注释文件
        if os.path.exists(annotation_path_src):
            shutil.move(annotation_path_src, annotation_path_dest)
            print(f"Moved annotation: {annotation_name} -> {annotation_dest}")


# 示例用法
if __name__ == "__main__":
    # 图片和注释的路径
    image_validation = "/wangyunhao/cds/wangyunhao/data/BTR/mmsegform/data1209/images/validation"
    image_training = "/wangyunhao/cds/wangyunhao/data/BTR/mmsegform/data1209/images/training"
    annotation_validation = "/wangyunhao/cds/wangyunhao/data/BTR/mmsegform/data1209/annotations/validation"
    annotation_training = "/wangyunhao/cds/wangyunhao/data/BTR/mmsegform/data1209/annotations/training"
    
    # 要移动的图片数量
    num_to_move = 30000
    
    # 执行移动
    move_images_and_annotations(image_validation, image_training, annotation_validation, annotation_training, num_to_move)
