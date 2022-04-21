# Human3d-keypoints

train
1、download datasets
data_3d_h36m.npz
data_2d_h36m_cpn_ft_h36m_dbb.npz
you can find it in https://github.com/Vegetebird/MHFormer

2、arc = 3,3,3,train video sequence nums=27,eg:
   python run.py -e 80 -k cpn_ft_h36m_dbb -arc 3,3,3
   python run.py -e 80 -k cpn_ft_h36m_dbb -arc 3,3,3,3
   python run.py -e 80 -k cpn_ft_h36m_dbb -arc 3,3,3,3,3

3、camera or video
   download checkpoints:https://drive.google.com/drive/folders/1TinLUEQ8C0hbyy0T7eMMgc-V0etz-m1s?usp=sharing
   python main.py
   
The project is developed based on VideoPose3d(https://github.com/facebookresearch/VideoPose3D).Combines relatively key points and trajectories to get global 3D key points，Thanks to Viedopose3d researchers.
