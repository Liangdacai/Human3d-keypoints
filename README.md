# Human3d-keypoints

1、download datasets  <br>
data_3d_h36m.npz  <br>
data_2d_h36m_cpn_ft_h36m_dbb.npz  <br>
you can find it in https://github.com/Vegetebird/MHFormer  <br>

2、training  <br>
   arc = 3,3,3,train video sequence nums=27,eg:  <br>
   python run.py -e 80 -k cpn_ft_h36m_dbb -arc 3,3,3  <br>
   python run.py -e 80 -k cpn_ft_h36m_dbb -arc 3,3,3,3  <br>
   python run.py -e 80 -k cpn_ft_h36m_dbb -arc 3,3,3,3,3  <br>

3、camera or video  <br>
   download checkpoints:https://drive.google.com/drive/folders/1TinLUEQ8C0hbyy0T7eMMgc-V0etz-m1s?usp=sharing  <br>
   python main.py  <br>
4、demo for camera:  <br>
   https://www.bilibili.com/video/BV1a44y1G7Jr?spm_id_from=333.999.0.0  <br>
   https://www.bilibili.com/video/BV1JA411G7jn?spm_id_from=333.999.0.0  <br> 
   
The project is developed based on VideoPose3d(https://github.com/facebookresearch/VideoPose3D).Combines relatively key points and trajectories to get global 3D key points，Thanks to Viedopose3d researchers.
