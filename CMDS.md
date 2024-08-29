# infer

python demos/demo_detect.py 

# data

rm /run/user/1000/gvfs/smb-share:server=172.20.254.132,share=sharedfolder/test/fire/labels/bosh -rvf
cp /home/manu/tmp/samples_pick/labels /run/user/1000/gvfs/smb-share:server=172.20.254.132,share=sharedfolder/test/fire/labels/bosh
rm /run/user/1000/gvfs/smb-share:server=172.20.254.132,share=sharedfolder/test/fire/images/bosh -rvf
cp /home/manu/tmp/samples_pick/images /run/user/1000/gvfs/smb-share:server=172.20.254.132,share=sharedfolder/test/fire/images/bosh -rvf