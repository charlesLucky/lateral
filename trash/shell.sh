
### test DS
python prepare_ds.py --stage 1 --ds_remark SESSION_TENT_NEW
python Train_DS.py --cfg_path ./configs/MDCMrect_1st.yaml --mode eager_tf   --batch_size 40
python prepare_ds.py --stage 2 --ds_remark SESSION_TENT_NEW
python Train_DS_2ed.py --cfg_path ./configs/MDCMrect_2ed_stage.yaml --stage 2  --batch_size 40
