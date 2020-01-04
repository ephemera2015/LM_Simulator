import glob
import os
import sys
import json
from io import BytesIO
import base64
import numpy as np
import requests
import tarfile
from tiffio.baseio import open_tif, save_tif

LOGFILE='./result.txt'
PADDING=20
PADDING_MODE='zero'
DATASET='/home/csdl/deli/neutu/data/gold/test'

config_template={
'model_path':'/home/csdl/deli/neutu/light/train_ffn_refined/model.ckpt-64425',
'segmentation_threshold':0.95,
'move_threshold':0.95
}


def generate_seeds(img):
    #return example[(z1,y1,x1),(z2,y2,x2),...,[(zn,yn,xn)]]
    seeds=[]
    depth,height,width=img.shape
    for k in range(0,depth-20,20):
        for j in range(0,height-40,40):
            for i in range(0,width-40,40):
                patch=img[k:k+20,j:j+40,i:i+40]
                if patch.max()>80:
                    idx=int(patch.argmax())
                    z=idx//1600
                    y=(idx-z*1600)//width
                    x=idx-z*1600-y*20
                    seeds.append([k+z,j+y,i+x])
    return seeds

def openSingleFile(file_name):
    rv={}
    f = tarfile.open(file_name,'r:gz')
    tif_names = f.getnames()
    for name in tif_names:
        if name.endswith('.swc.tif'):
            f.extract(name,'./')
            label = open_tif(name)
            os.remove(name)
        elif name.endswith('.tif'):
            f.extract(name,'./')
            img = open_tif(name)
            os.remove(name)
    return img,label
        

def loadTestDataset(dataset):
    file_names, imgs, labels = [],[],[]
    for file_name in glob.glob(dataset+'/*.tar.gz'):
        file_names.append(file_name)
        img,label = openSingleFile(file_name)
        imgs.append(img)
        labels.append(label)
    return file_names, imgs,labels

def segment(img,service_url):
    if not service_url.endswith('/'):
        service_url+='/'

    sz=img.shape[0]+2*PADDING
    sy=img.shape[1]+2*PADDING
    sx=img.shape[2]+2*PADDING

    mean=np.mean(img)
    std=np.std(img)

    if PADDING_MODE=='zero':
        _img=np.zeros((img.shape[0]+2*PADDING,img.shape[1]+2*PADDING,img.shape[2]+2*PADDING),dtype=img.dtype)
    elif PADDING_MODE=='gauss':
        _img=np.random.gauss(size=(sz,sy,sx),mean=mean,std=std)
    _img[PADDING:-PADDING,PADDING:-PADDING,PADDING:-PADDING]=img

    buf=BytesIO()
    np.savez_compressed(buf,_img)
    img_encoded=base64.b64encode(buf.getvalue()).decode('utf-8')
    config=config_template
    config['img']=img_encoded
    config['seeds']=generate_seeds(img)
    
    r=requests.post(service_url+'light_seg',data=json.dumps(config),headers={'content-type':'application/json'})
    with open('tmp.npz','wb') as f:
        f.write(r.content)
    data = np.load('tmp.npz')
    os.remove("tmp.npz")
    _rv=data['segmentation'].astype(np.uint8)
    _rv[_rv>0]=1
    _rv[_img==0]=0
    return _rv[PADDING:-PADDING,PADDING:-PADDING,PADDING:-PADDING]

def compare(segmentation,ground_truth):
    total_cnt = segmentation.size
    true_pos ,true_neg = 0, 0 
    false_pos , false_neg = 0, 0

    for a,b in zip(np.nditer(segmentation),np.nditer(ground_truth)):
        if a and b:
            true_pos += 1
        elif (not a) and (not b):
            true_neg += 1
        elif a and (not b):
            false_pos += 1
        elif (not a) and b:
            false_neg += 1
    assert(total_cnt == true_pos + true_neg + false_pos+ false_neg)
    rv={}
    rv['accuracy'] = (true_pos+true_neg)/(total_cnt)
    rv['recall'] = true_pos/(true_pos+false_neg+1e-10)
    rv['presision'] = true_pos/(true_pos+false_pos+1e-10)
    rv['f1'] = 2*rv['recall']*rv['presision']/(rv['recall']+rv['presision']+1e-10)
    return rv

def evalLightFFNModel(dataset,service_url,save_segmentation=False):
    rv={}
    file_names, imgs, labels = loadTestDataset(dataset)
    with open(LOGFILE,'w') as f:
        for file_name, img, label in zip(file_names,imgs,labels):
            print('#processing {file_name}'.format(file_name=file_name))
            segmentation = segment(img,service_url)
            file_name=os.path.split(file_name)[1]
            if save_segmentation:
                save_tif(segmentation,file_name+'_seg.tif')
            result=compare(segmentation,label)
            rv[file_name]=result
        f.write(json.dumps(rv))
    return rv


if __name__ == '__main__':
    print(evalLightFFNModel(dataset=DATASET,
                            service_url=r'http://10.14.111.154:4321',save_segmentation=True))
