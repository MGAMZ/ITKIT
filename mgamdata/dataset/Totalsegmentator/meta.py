CLASS_INDEX_MAP = {
    'background': 0, # 背景
    'adrenal_gland_left': 1, # 左肾上腺
    'adrenal_gland_right': 2, # 右肾上腺
    'aorta': 3, # 主动脉
    'atrial_appendage_left': 4, # 左心耳
    'autochthon_left': 5, # 左自体骨
    'autochthon_right': 6, # 右自体骨
    'brachiocephalic_trunk': 7, # 头臂干
    'brachiocephalic_vein_left': 8, # 左头臂静脉
    'brachiocephalic_vein_right': 9, # 右头臂静脉
    'brain': 10, # 脑
    'clavicula_left': 11, # 左锁骨
    'clavicula_right': 12, # 右锁骨
    'colon': 13, # 结肠
    'common_carotid_artery_left': 14, # 左颈总动脉
    'common_carotid_artery_right': 15, # 右颈总动脉
    'costal_cartilages': 16, # 肋软骨
    'ct': 17, # 计算机断层扫描（此项似乎为缩写，通常意指影像类型而非解剖结构）
    'duodenum': 18, # 十二指肠
    'esophagus': 19, # 食管
    'femur_left': 20, # 左股骨
    'femur_right': 21, # 右股骨
    'gallbladder': 22, # 胆囊
    'gluteus_maximus_left': 23, # 左臀大肌
    'gluteus_maximus_right': 24, # 右臀大肌
    'gluteus_medius_left': 25, # 左臀中肌
    'gluteus_medius_right': 26, # 右臀中肌
    'gluteus_minimus_left': 27, # 左臀小肌
    'gluteus_minimus_right': 28, # 右臀小肌
    'heart': 29, # 心脏
    'hip_left': 30, # 左髋
    'hip_right': 31, # 右髋
    'humerus_left': 32, # 左肱骨
    'humerus_right': 33, # 右肱骨
    'iliac_artery_left': 34, # 左髂动脉
    'iliac_artery_right': 35, # 右髂动脉
    'iliac_vena_left': 36, # 左髂静脉
    'iliac_vena_right': 37, # 右髂静脉
    'iliopsoas_left': 38, # 左髂腰肌
    'iliopsoas_right': 39, # 右髂腰肌
    'inferior_vena_cava': 40, # 下腔静脉
    'kidney_cyst_left': 41, # 左肾囊肿
    'kidney_cyst_right': 42, # 右肾囊肿
    'kidney_left': 43, # 左肾
    'kidney_right': 44, # 右肾
    'liver': 45, # 肝
    'lung_lower_lobe_left': 46, # 左肺下叶
    'lung_lower_lobe_right': 47, # 右肺下叶
    'lung_middle_lobe_right': 48, # 右肺中叶
    'lung_upper_lobe_left': 49, # 左肺上叶
    'lung_upper_lobe_right': 50, # 右肺上叶
    'pancreas': 51, # 胰腺
    'portal_vein_and_splenic_vein': 52, # 门静脉和脾静脉
    'prostate': 53, # 前列腺
    'pulmonary_vein': 54, # 肺静脉
    'rib_left_1': 55, # 左第1肋骨
    'rib_left_10': 56, # 左第10肋骨
    'rib_left_11': 57, # 左第11肋骨
    'rib_left_12': 58, # 左第12肋骨
    'rib_left_2': 59, # 左第2肋骨
    'rib_left_3': 60, # 左第3肋骨
    'rib_left_4': 61, # 左第4肋骨
    'rib_left_5': 62, # 左第5肋骨
    'rib_left_6': 63, # 左第6肋骨
    'rib_left_7': 64, # 左第7肋骨
    'rib_left_8': 65, # 左第8肋骨
    'rib_left_9': 66, # 左第9肋骨
    'rib_right_1': 67, # 右第1肋骨
    'rib_right_10': 68, # 右第10肋骨
    'rib_right_11': 69, # 右第11肋骨
    'rib_right_12': 70, # 右第12肋骨
    'rib_right_2': 71, # 右第2肋骨
    'rib_right_3': 72, # 右第3肋骨
    'rib_right_4': 73, # 右第4肋骨
    'rib_right_5': 74, # 右第5肋骨
    'rib_right_6': 75, # 右第6肋骨
    'rib_right_7': 76, # 右第7肋骨
    'rib_right_8': 77, # 右第8肋骨
    'rib_right_9': 78, # 右第9肋骨
    'sacrum': 79, # 骶骨
    'scapula_left': 80, # 左肩胛骨
    'scapula_right': 81, # 右肩胛骨
    'skull': 82, # 颅骨
    'small_bowel': 83, # 小肠
    'spinal_cord': 84, # 脊髓
    'spleen': 85, # 脾
    'sternum': 86, # 胸骨
    'stomach': 87, # 胃
    'subclavian_artery_left': 88, # 左锁骨下动脉
    'subclavian_artery_right': 89, # 右锁骨下动脉
    'superior_vena_cava': 90, # 上腔静脉
    'thyroid_gland': 91, # 甲状腺
    'trachea': 92, # 气管
    'urinary_bladder': 93, # 膀胱
    'vertebrae_C1': 94, # 第1颈椎
    'vertebrae_C2': 95, # 第2颈椎
    'vertebrae_C3': 96, # 第3颈椎
    'vertebrae_C4': 97, # 第4颈椎
    'vertebrae_C5': 98, # 第5颈椎
    'vertebrae_C6': 99, # 第6颈椎
    'vertebrae_C7': 100, # 第7颈椎
    'vertebrae_L1': 101, # 第1腰椎
    'vertebrae_L2': 102, # 第2腰椎
    'vertebrae_L3': 103, # 第3腰椎
    'vertebrae_L4': 104, # 第4腰椎
    'vertebrae_L5': 105, # 第5腰椎
    'vertebrae_S1': 106, # 第1骶椎
    'vertebrae_T1': 107, # 第1胸椎
    'vertebrae_T10': 108, # 第10胸椎
    'vertebrae_T11': 109, # 第11胸椎
    'vertebrae_T12': 110, # 第12胸椎
    'vertebrae_T2': 111, # 第2胸椎
    'vertebrae_T3': 112, # 第3胸椎
    'vertebrae_T4': 113, # 第4胸椎
    'vertebrae_T5': 114, # 第5胸椎
    'vertebrae_T6': 115, # 第6胸椎
    'vertebrae_T7': 116, # 第7胸椎
    'vertebrae_T8': 117, # 第8胸椎
    'vertebrae_T9': 118, # 第9胸椎
}

SUBSETS = {
    'vertebral': [
        'background',
        'rib_left_1', 'rib_left_2', 'rib_left_3', 'rib_left_4', 'rib_left_5',
        'rib_left_6', 'rib_left_7', 'rib_left_8', 'rib_left_9', 
        'rib_left_10', 'rib_left_11', 'rib_left_12',
        'rib_right_1', 'rib_right_2', 'rib_right_3', 'rib_right_4', 'rib_right_5',
        'rib_right_6', 'rib_right_7', 'rib_right_8', 'rib_right_9',
        'rib_right_10', 'rib_right_11', 'rib_right_12',
        'sacrum', 
        'vertebrae_C1', 'vertebrae_C2', 'vertebrae_C3', 'vertebrae_C4',
        'vertebrae_C5', 'vertebrae_C6', 'vertebrae_C7', 
        'vertebrae_T1', 'vertebrae_T2', 'vertebrae_T3', 'vertebrae_T4',
        'vertebrae_T5', 'vertebrae_T6', 'vertebrae_T7', 'vertebrae_T8',
        'vertebrae_T9', 'vertebrae_T10', 'vertebrae_T11',
        'vertebrae_T12', 
        'vertebrae_L1', 'vertebrae_L2', 'vertebrae_L3', 'vertebrae_L4', 'vertebrae_L5',
        'vertebrae_S1', 
    ]
}

CLASS_MERGE = {
    'general': {
        'adrenal_gland': [
            'adrenal_gland_left', 'adrenal_gland_right'
        ],
        'autochthon': [
            'autochthon_left', 'autochthon_right'
        ],
        'brachiocephalic': [
            'brachiocephalic_trunk', 'brachiocephalic_vein_left',
            'brachiocephalic_vein_right'
        ],
        'clavicula': [
            'clavicula_left', 'clavicula_right'
        ],
        'common_carotid_artery': [
            'common_carotid_artery_left', 'common_carotid_artery_right'
        ],
        'femur': [
            'femur_left', 'femur_right'
        ],
        'gluteus': [
            'gluteus_maximus_left', 'gluteus_maximus_right',
            'gluteus_medius_left', 'gluteus_medius_right',
            'gluteus_minimus_left', 'gluteus_minimus_right'
        ],
        'hip': [
            'hip_left', 'hip_right'
        ],
        'humerus': [
            'humerus_left', 'humerus_right'
        ],
        'iliac': [
            'iliac_artery_left', 'iliac_artery_right', 'iliac_vena_left',
            'iliac_vena_right'
        ],
        'iliopsoas': [
            'iliopsoas_left', 'iliopsoas_right'
        ],
        'kidney': [
            'kidney_cyst_left', 'kidney_cyst_right', 'kidney_left',
            'kidney_right'
        ],
        'lung': [
            'lung_lower_lobe_left', 'lung_lower_lobe_right',
            'lung_middle_lobe_right', 'lung_upper_lobe_left',
            'lung_upper_lobe_right'
        ],
        'rib': [
            'rib_left_1', 'rib_left_10', 'rib_left_11', 'rib_left_12',
            'rib_left_2', 'rib_left_3', 'rib_left_4', 'rib_left_5',
            'rib_left_6', 'rib_left_7', 'rib_left_8', 'rib_left_9',
            'rib_right_1', 'rib_right_10', 'rib_right_11', 'rib_right_12',
            'rib_right_2', 'rib_right_3', 'rib_right_4', 'rib_right_5',
            'rib_right_6', 'rib_right_7', 'rib_right_8', 'rib_right_9'
        ],
        'scapula': [
            'scapula_left', 'scapula_right'
        ],
        'subclavian_artery': [
            'subclavian_artery_left', 'subclavian_artery_right'
        ],
        'vertebrae': [
            'vertebrae_C1', 'vertebrae_C2', 'vertebrae_C3', 'vertebrae_C4',
            'vertebrae_C5', 'vertebrae_C6', 'vertebrae_C7', 'vertebrae_L1',
            'vertebrae_L2', 'vertebrae_L3', 'vertebrae_L4', 'vertebrae_L5',
            'vertebrae_S1', 'vertebrae_T1', 'vertebrae_T10', 'vertebrae_T11',
            'vertebrae_T12', 'vertebrae_T2', 'vertebrae_T3', 'vertebrae_T4',
            'vertebrae_T5', 'vertebrae_T6', 'vertebrae_T7', 'vertebrae_T8',
            'vertebrae_T9'
        ]
    }
}


def generate_subset_class_map_and_label_map(subset_name: str):
    """
    生成从原始类定义到子集类定义的映射字典
    
    Args:
        subset_name: 子集名称，在SUBSETS字典中定义
        
    Returns:
        tuple: (subset_class_map, label_map)
            - subset_class_map: 子集类名到新索引的映射 {class_name: new_index}
            - label_map: 原始类索引到新索引的映射 {old_index: new_index}
    """
    if subset_name not in SUBSETS:
        raise ValueError(f"Subset '{subset_name}' not found in SUBSETS. Available subsets: {list(SUBSETS.keys())}")
    
    subset_classes = SUBSETS[subset_name]
    if not subset_classes:
        # 如果子集为空，返回原始映射
        return CLASS_INDEX_MAP.copy(), {v: v for v in CLASS_INDEX_MAP.values()}
    
    # 创建子集类名到新索引的映射
    subset_class_map = {class_name: idx for idx, class_name in enumerate(subset_classes)}
    
    # 创建原始类索引到新索引的映射
    label_map = {}
    for class_name, old_index in CLASS_INDEX_MAP.items():
        if class_name in subset_class_map:
            # 如果类在子集中，映射到新索引
            label_map[old_index] = subset_class_map[class_name]
        else:
            # 如果类不在子集中，映射到background (0)
            label_map[old_index] = 0
    
    return subset_class_map, label_map


def generate_reduced_class_map_and_label_map(reduction):
    """
    根据原始CLASS_INDEX_MAP和REDUCTION，生成合并后的CLASS_MAP和label_map。

    Args:
        class_index_map (dict): 原始类别名到id的映射，如 {"cat": 0, "dog": 1, ...}
        reduction (dict): 合并后的类组名及其包含的源类名，如 {"animal": ["cat", "dog"], ...}

    Returns:
        reduced_class_map (dict): 合并后类别名到新id的映射，如 {"animal": 0, ...}
        label_map (dict): 旧id到新id的映射，如 {0: 0, 1: 0, ...}
    """
    # 1. 构建源类名到合并后类名的映射
    source_to_group = {}
    for group, sources in reduction.items():
        for src in sources:
            source_to_group[src] = group

    # 2. 新类别名集合（保留未被合并的类别）
    all_group_names = set(reduction.keys())
    for name in CLASS_INDEX_MAP:
        if name not in source_to_group:
            all_group_names.add(name)
    reduced_class_names = sorted(list(all_group_names))
    reduced_class_names.remove('background')
    reduced_class_names.insert(0, 'background')
    reduced_class_map = {name: idx for idx, name in enumerate(reduced_class_names)}

    # 3. 构建label_map
    label_map = {}
    for name, old_id in CLASS_INDEX_MAP.items():
        group_name = source_to_group.get(name, name)
        new_id = reduced_class_map[group_name]
        label_map[old_id] = new_id

    return reduced_class_map, label_map
