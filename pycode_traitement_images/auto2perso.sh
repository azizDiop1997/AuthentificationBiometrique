echo "preprocessing running"

python3 traitement_imagesv1.py -i ../BDD_FingerVeins/001left_index_1.bmp -o out_11.bmp
python3 traitement_imagesv1.py -i ../BDD_FingerVeins/001left_index_2.bmp -o out_12.bmp
python3 traitement_imagesv1.py -i ../BDD_FingerVeins/001left_index_3.bmp -o out_13.bmp
python3 traitement_imagesv1.py -i ../BDD_FingerVeins/002left_index_4.bmp -o out_21.bmp
python3 traitement_imagesv1.py -i ../BDD_FingerVeins/002left_index_5.bmp -o out_22.bmp
python3 traitement_imagesv1.py -i ../BDD_FingerVeins/002left_index_6.bmp -o out_23.bmp

python3 contrastEnhance.py -i out_11.bmp -o 11.bmp
python3 contrastEnhance.py -i out_12.bmp -o 12.bmp
python3 contrastEnhance.py -i out_13.bmp -o 13.bmp
python3 contrastEnhance.py -i out_21.bmp -o 21.bmp
python3 contrastEnhance.py -i out_22.bmp -o 22.bmp
python3 contrastEnhance.py -i out_23.bmp -o 23.bmp

rm out_*

echo "finished - training set ready"
