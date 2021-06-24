import sys
sys.path.append(r'/home/roncax/Git/Pytorch-UNet/') # /content/gdrive/MyDrive/Colab/Thesis_OaR_Segmentation/

import OaR_segmentation.utilities.paths as paths
import structseg2019_load

if __name__ == '__main__':
    structseg2019_load.prepare_structseg(paths=paths.Paths(db="StructSeg2019_Task3_Thoracic_OAR"))
