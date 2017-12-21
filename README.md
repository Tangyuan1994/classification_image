# training_model_VGGface(Pour mon dossier vggface)
1.  copy ce dossier vggface et le ficher vggface.prototxt dans le dossier caffe 
(Remarque: Le fichier vggface.prototxt doit être en dehors du dossier vggface,vous devez aussi ajouter le VGG_FACE.caffemodel par vous-même dans mon dossier vggface)
2.  cd caffe
3.  sh vggface/create-lmdb.sh
4.  build/tools/compute_image_mean vggface/train_lmdb vggface/mean.binaryproto
5.  ./build/tools/caffe train \
    --solver=vggface/solver.prototxt \
    --weights=vggface/VGG_FACE.caffemodel



# classification_image_HF(Pour mon dossier modelHF)
<p> 1.Download img_align_celeba.zip and list_attr_celeba.txt.</p>
2.Delete the first line of list_attr_celeba.txt.(If we keep the first line it will lead the error of index out of range beacause we couldn't read the 21th colone(attribute "male") of the first line.)
3.Put the 5 files (gender_net.caffemodel,deploy_gender.prototxt,mean.binaryproto,list_attr_celeba.txt,model.py) in the directory img_align_celeba.
4.run model.py (python model.py)
5.You will find a new file called prediction.txt, in this file you will find all the prediction of label orderly(the first is the label of 000001.jpg, the seconde is the label of 000002.jpg,the third is the label of 000003.jpg)
6.You will find the accuracy in the terminal.
# classification_image
