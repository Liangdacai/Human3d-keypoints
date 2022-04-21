# Human3d-keypoints

train
1、load datasets
data_3d_h36m.npz
data_2d_h36m_cpn_ft_h36m_dbb.npz
you can find it in https://github.com/Vegetebird/MHFormer

2、arc = 3,3,3,train video sequence nums=27
   python run.py -e 80 -k cpn_ft_h36m_dbb -arc 3,3,3
   python run.py -e 80 -k cpn_ft_h36m_dbb -arc 3,3,3,3
   python run.py -e 80 -k cpn_ft_h36m_dbb -arc 3,3,3,3,3
