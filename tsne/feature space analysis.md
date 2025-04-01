<p align="center">
    <img src="https://github.com/ZHOURui6025/COSDA-master/blob/master/tsne/OfficeHome_a_0_b_3_osbp_00.png" width="32%">
    <img src="https://github.com/ZHOURui6025/COSDA-master/blob/master/tsne/OfficeHome_a_0_b_3_anna_00.png" width="32%">
    <img src="https://github.com/ZHOURui6025/COSDA-master/blob/master/tsne/OfficeHome_a_0_b_3_cosda_00.png" width="32%">
</p>
<br>
<p align="center">
    <img src="https://github.com/ZHOURui6025/COSDA-master/blob/master/tsne/Office_a_2_b_1_osbp_00.png" width="32%">
    <img src="https://github.com/ZHOURui6025/COSDA-master/blob/master/tsne/Office_a_2_b_1_anna_00.png" width="32%">
    <img src="https://github.com/ZHOURui6025/COSDA-master/blob/master/tsne/Office_a_2_b_1_cosda_00.png" width="32%">
</p>
Fig. 2 The t-SNE visualization of feature distributions on the $W \rightarrow D$ task (\textit{Office-31}, left) and $Ar \rightarrow Rw$ task (\textit{Office-Home}, right)  with the ResNet-50 backbone. Comparative methods include OSBP, ANNA and COSDA. The gray node denotes the unknown target sample, the red node denotes the known source sample, and the blue node denotes the known target sample. Compared with OSBP, our method clusters class characteristics more compactly, which imdicates the improvement of the decision border between known and unknown classes. Since both OSBP and ANNA utilize adversarial training and share the same underlying framework, their feature spaces exhibit similar characteristics. Comparatively, ANNA better delineates the boundaries between known and unknown classes, as only a small number of gray points overlap with the blue and red points. Due to their similar performance, we cannot clearly exhibit COSDAis advantage at the feature level comparing with ANNA. However, the clustering tendency of local unknown classes suggests that VMP’s approach of considering unknown classes as multiple distinct groups is beneficial.
