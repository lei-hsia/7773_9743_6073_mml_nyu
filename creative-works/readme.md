PCA & Manifold的
1. base theory: なぜ実際にn次元の次元データを2次元の次元データに縮小できますか？
                为什么实际上可以把 n-D dimensional data 降为 2-D dimension的数据? 
                
             关键: Topology!!
             Topology: A particular way of connecting points & space.==> space can be FINITE BUT UNBOUNDED.
             e.g. 2-D 平面: 折为 3D cylinder: vertical:还是finite&&bounded; 
                                         horizontal: unbounded
                           折为 3D sphere: 这样从任意一个方向就都是 unbounded, 但是这个sphere的本质还是平面: finite
                           这种 本来不是平面上面的东西，如果强行画在plane上面：造成distortion
                           很好的epitome: 
                           ニューヨークから東京までの飛行機は、飛行機のボード上を見ると常にカーブを描いていますが、
                           3D空間に置かれたラインはありません; 地球上如果从纽约飞到東京, 飞机上面小屏幕上面画的航线
                           总是偏向北极方向弯曲的，但是实际上如果从3D的球体表面看, 那条线就是这两点之间的最短距离
                           
             从topology的角度看，2D的平面确实可以升维变为3D，那么理论上就可以升到更高维变为n维;
             每次升降维度都会有distortion: 因为这个topological space是 Non-Euclidean的
             
2. 和PCA/Manifold的关系: 正是因为有了这个base theory, PCA 采取的降维才有意义，否则没有这样做makes sense的理论基础

3. PCA获取PCs的具体做法: 
    1. 所有的点本来都在第一象限
    2. PCA本质上是线性变换(包括Eigenvalue Decomposition, SVD), linear的话, (mean_x, mean_y)这个点是一定会通过的;
    3. 移动这个mean的点到origin, 所有的sample点的****相对位置****不变
    4. 最小化各点到直线的垂直距离的平方的和: 因为mean点到各点的相对位置不变, 
        由勾股定理, "最小化各点到直线的垂直距离的平方的和" <=> 最大化各点到origin的距离的平方和; 
    5. 这条直线所确定的方向，实际上就是Eigenvector对应的方向                              ************
    6. 这个平方和Sum of Squared Distances(SSD): 实际上就是这个eigenvector对应的Eigenvalue **********
    7. 这条直线, 就是一个Principal Component (PC1)                                    ************
    8. 比如有两个features, 确定的直线的斜率是0.25, i.e. x轴对应的feature每变换4个单位，
        y轴对应的feature才变化1个单位:  x轴相对y轴更能代表这个datasets的各种变化
    9. squareRoot(eigenvalue for PC1) = Singular Value for PC1
    10. 类似RF的feature_importance_画出来的PC的重要性的图叫scree plot
    11. 所有其他的PC也都是这样画出来的, 对n维数据，确定方向后, 要确保每一个画出来的直线都是垂直于之前的所有的直线
    12. 实际上对现实中的数据，由于事物的非对称性，很多时候2D就很能代表整个dataset
    13. Variation for PC1 = SSD(PC1)/(n-1):  variation是PC对数据的另一种代表性的视角
               
