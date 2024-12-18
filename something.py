import random
import torch
import logging

# Hàm tấn công vào mô hình (thêm nhiễu vào trọng số của model net)
def apply_poison_attack(net, node_id, device, noise_factor=0.1):
    
    with torch.no_grad():
        net.eval()  # Đảm bảo mô hình không thay đổi trong quá trình tấn công
        # Lấy trọng số của mô hình (target model)
        for param in net.parameters():
            param += torch.randn_like(param) * noise_factor  # Thêm nhiễu ngẫu nhiên vào trọng số

        logging.info(f"Poison attack applied on model (net) at node {node_id} with modified weights.")
    return net

#MPAF
import torch
import numpy as np
import copy 

def create_random_base_model(model): 
    """
    Tạo một mô hình base bằng cách làm nhiễu nhẹ các trọng số của model hiện tại.
    """
    base_model = copy.deepcopy(model)
    for param in base_model:
        
        base_model[param] += torch.randn_like(base_model[param]) * 0.1  # Small random perturbation
        
    return base_model

def create_fake_client_update(global_weights, base_weights, scaling_factor):
    """
    Tạo ra cập nhật giả sử dụng sự khác biệt giữa mô hình toàn cầu và mô hình base.
    """
    fake_update = {}

    for key in global_weights.keys():
        fake_update[key] = (base_weights[key] - global_weights[key]) * scaling_factor

    return fake_update

def apply_poison_attack(global_model, base_model, attack_rate, num_nodes, device):
    """
    Tạo danh sách cập nhật giả (fake updates) cho các node bị tấn công.
    """
    num_nodes_to_attack = int(num_nodes * attack_rate)
    fake_nodes = torch.randperm(num_nodes)[:num_nodes_to_attack]
    
    global_weights = global_model.state_dict() 
    fake_updates = []  # Lưu các cập nhật giả

    for node_id in fake_nodes:
        #base_model = create_random_base_model(target_model)
        fake_update = create_fake_client_update(global_weights, base_model, scaling_factor=1e6)
        fake_updates.append(fake_update)
    
    return fake_updates

