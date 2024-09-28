# infer

export PYTHONPATH=$PYTHONPATH:$(pwd)

screen python demos/demo_detect.py 

python demos/demo_detect_mp.py --show True --save_root /home/manu/tmp/fire_test_results_single

# data

rm /run/user/1000/gvfs/smb-share:server=172.20.254.132,share=sharedfolder/test/fire/labels/bosh -rvf
cp /home/manu/tmp/samples_pick/labels /run/user/1000/gvfs/smb-share:server=172.20.254.132,share=sharedfolder/test/fire/labels/bosh
rm /run/user/1000/gvfs/smb-share:server=172.20.254.132,share=sharedfolder/test/fire/images/bosh -rvf
cp /home/manu/tmp/samples_pick/images /run/user/1000/gvfs/smb-share:server=172.20.254.132,share=sharedfolder/test/fire/images/bosh -rvf