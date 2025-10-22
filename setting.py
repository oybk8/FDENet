setting ={
    'exper_name':'FDENet',# 模型-训练集 SwinT_OA_JL_RN_AS Transx_XR
    'crop_size':(512,512),#(1024,1024)#1280，1024 #1472   (512,512)
    'dataset':'JLYH', # 'DPGlobe' or 'JHWV2' or'Mass'
    'dataroot':'./dataset/JLYH', #
    'model':'FDENet', 
    'batch_size_card': 4,
    'gpu':'3',
    'seed':197,
    'epoch':150,
    'out_pred':True,
    'remarks':'',
}

