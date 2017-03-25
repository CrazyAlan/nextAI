#Install anaconda 


#Train and save feature 
python src/align/train_person_recognition_with_dlib.py --input_dir ~/remote/datasets/lfw/lfw_svm_10_raw --output_dir ~/remote/datasets/lfw/lfw_mtcnnpy_128_raw --image_size 128 --margin 24 --random_order --model_dir /home/xca64/remote/models/facenet/lightened/20170321-222346 --feature_dir /home/xca64/remote/feature/lfw_svm_10 --feature_name emb_array

#Train SVM 
python util/train_svm.py --npz_file_dir /home/xca64/remote/feature/lfw_svm_10/emb_array.npz --C 4 --svm_model_dir ~/remote/models/svm --gamma 1 

#Detect and Tracking people
python src/align/detect_tracking_recognize.py --input_dir /home/xca64/remote/datasets/lfw/test --image_size 128 --margin 24 --model_dir /home/xca64/remote/models/facenet/lightened/20170321-222346 --svm_model_dir ~/remote/models/svm

