CLASS_INDEX_MAP = {
    'background': 0,
    'adrenal_gland_left': 1,
    'adrenal_gland_right': 2,
    'aorta': 3,
    'atrial_appendage_left': 4,
    'autochthon_left': 5,
    'autochthon_right': 6,
    'brachiocephalic_trunk': 7,
    'brachiocephalic_vein_left': 8,
    'brachiocephalic_vein_right': 9,
    'brain': 10,
    'clavicula_left': 11,
    'clavicula_right': 12,
    'colon': 13,
    'common_carotid_artery_left': 14,
    'common_carotid_artery_right': 15,
    'costal_cartilages': 16,
    'ct': 17,
    'duodenum': 18,
    'esophagus': 19,
    'femur_left': 20,
    'femur_right': 21,
    'gallbladder': 22,
    'gluteus_maximus_left': 23,
    'gluteus_maximus_right': 24,
    'gluteus_medius_left': 25,
    'gluteus_medius_right': 26,
    'gluteus_minimus_left': 27,
    'gluteus_minimus_right': 28,
    'heart': 29,
    'hip_left': 30,
    'hip_right': 31,
    'humerus_left': 32,
    'humerus_right': 33,
    'iliac_artery_left': 34,
    'iliac_artery_right': 35,
    'iliac_vena_left': 36,
    'iliac_vena_right': 37,
    'iliopsoas_left': 38,
    'iliopsoas_right': 39,
    'inferior_vena_cava': 40,
    'kidney_cyst_left': 41,
    'kidney_cyst_right': 42,
    'kidney_left': 43,
    'kidney_right': 44,
    'liver': 45,
    'lung_lower_lobe_left': 46,
    'lung_lower_lobe_right': 47,
    'lung_middle_lobe_right': 48,
    'lung_upper_lobe_left': 49,
    'lung_upper_lobe_right': 50,
    'pancreas': 51,
    'portal_vein_and_splenic_vein': 52,
    'prostate': 53,
    'pulmonary_vein': 54,
    'rib_left_1': 55,
    'rib_left_10': 56,
    'rib_left_11': 57,
    'rib_left_12': 58,
    'rib_left_2': 59,
    'rib_left_3': 60,
    'rib_left_4': 61,
    'rib_left_5': 62,
    'rib_left_6': 63,
    'rib_left_7': 64,
    'rib_left_8': 65,
    'rib_left_9': 66,
    'rib_right_1': 67,
    'rib_right_10': 68,
    'rib_right_11': 69,
    'rib_right_12': 70,
    'rib_right_2': 71,
    'rib_right_3': 72,
    'rib_right_4': 73,
    'rib_right_5': 74,
    'rib_right_6': 75,
    'rib_right_7': 76,
    'rib_right_8': 77,
    'rib_right_9': 78,
    'sacrum': 79,
    'scapula_left': 80,
    'scapula_right': 81,
    'skull': 82,
    'small_bowel': 83,
    'spinal_cord': 84,
    'spleen': 85,
    'sternum': 86,
    'stomach': 87,
    'subclavian_artery_left': 88,
    'subclavian_artery_right': 89,
    'superior_vena_cava': 90,
    'thyroid_gland': 91,
    'trachea': 92,
    'urinary_bladder': 93,
    'vertebrae_C1': 94,
    'vertebrae_C2': 95,
    'vertebrae_C3': 96,
    'vertebrae_C4': 97,
    'vertebrae_C5': 98,
    'vertebrae_C6': 99,
    'vertebrae_C7': 100,
    'vertebrae_L1': 101,
    'vertebrae_L2': 102,
    'vertebrae_L3': 103,
    'vertebrae_L4': 104,
    'vertebrae_L5': 105,
    'vertebrae_S1': 106,
    'vertebrae_T1': 107,
    'vertebrae_T10': 108,
    'vertebrae_T11': 109,
    'vertebrae_T12': 110,
    'vertebrae_T2': 111,
    'vertebrae_T3': 112,
    'vertebrae_T4': 113,
    'vertebrae_T5': 114,
    'vertebrae_T6': 115,
    'vertebrae_T7': 116,
    'vertebrae_T8': 117,
    'vertebrae_T9': 118,
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

GENERAL_REDUCTION = {
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
