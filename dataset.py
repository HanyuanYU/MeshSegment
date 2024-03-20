import numpy as np

import trimesh
import open3d as o3d
import os
from plyfile import PlyData, PlyElement
import pandas as pd

import random
from collections import deque

'''
1、 read mesh
'''
gt_train_mesh_dir='E:\MeshSementicSegment\mesh_gt_train.ply'
gt_test_mesh_dir='E:\MeshSementicSegment\mesh_gt_test.ply'

# 读取vertices 、 triangles、vertex_normals、triangle_normals、
gt_train_mesh=o3d.io.read_triangle_mesh(gt_train_mesh_dir)
gt_test_mesh=o3d.io.read_triangle_mesh(gt_test_mesh_dir)
# print('gt_train_mesh.vertices: ',np.array(gt_train_mesh.vertices))
# [[-20.5992     1.73001  -50.8325  ]
#  [-20.5903     1.74062  -50.8102  ]
#  [-20.5739     1.76608  -50.7897  ]
#  ...
#  [  9.93205   -0.51925   12.0153  ]
#  [  9.93276   -2.27112    9.95537 ]
#  [  9.95712   -0.524293  12.0146  ]]

'''使用plyfile库'''
# 读取faces_colors
mesh_gt_train = PlyData.read(gt_train_mesh_dir)  # 读取文件
mesh_gt_test = PlyData.read(gt_test_mesh_dir)  # 读取文件
# Vertices = mesh_gt_train.elements[0].data  # 读取数据
# print(Vertices)
faces_train= mesh_gt_train.elements[1].data
faces_test= mesh_gt_test.elements[1].data
# print(faces)

# Vertices_pd = pd.DataFrame(Vertices)  # 转换成DataFrame, 因为DataFrame可以解析结构化的数据
# Vertices_np = np.zeros(Vertices_pd.shape)  # 初始化储存数据的array
faces_pd = pd.DataFrame(faces_train)  # 转换成DataFrame, 因为DataFrame可以解析结构化的数据
faces_pd_test = pd.DataFrame(faces_test)
# print('faces_pd.shape:')
# print(faces_pd.shape)
# faces_np = np.zeros(faces_pd.shape,dtype=np.int64)
faces_num=faces_pd.shape[0]
faces_np = np.zeros([faces_num,6],dtype=np.int64)
faces_num_test=faces_pd_test.shape[0]
faces_np_test = np.zeros([faces_num_test,6],dtype=np.int64)

# Vertices_property = Vertices[0].dtype.names  # 读取property的名字
face_property= faces_train[0].dtype.names
face_property_test= faces_test[0].dtype.names
# print('Vertices_property:')
# print(Vertices_property)
# print('face_property:')
# print(face_property)

# 提取vertex相关属性并转换成np格式
# for i, name in enumerate(Vertices_property):  # 按property读取数据，这样可以保证读出的数据是同样的数据类型。
#     # print(i,'  ',name)
#     Vertices_np[:, i] = Vertices_pd[name]
# print('Vertices_np.shape:')
# print(Vertices_np.shape)

# 提取face相关属性并转换成np格式
# for i, name in enumerate(face_property):  # 按property读取数据，这样可以保证读出的数据是同样的数据类型。
#     print('i:',i,'  name:',name)
#     if name=='vertex_indices':
#         # print(faces_pd[name].values)
#         # faces_np[:, i:3]=np.asarray(faces_pd[name], dtype=object)
#         # face_V=np.asarray(faces_pd[name].values[:]).reshape(faces_num, 3)
#         # print('face_V:',face_V)
#         j=0
#         while j<faces_num:
#             faces_np[j, 0:3] = faces_pd[name].values[j]
#             j=j+1
#     else:
#         # 提取face_colors
#         faces_np[:, i+2] = faces_pd[name]
# print('faces_np:',faces_np)
# 提取face相关属性并转换成np格式
for i, name in enumerate(face_property_test):  # 按property读取数据，这样可以保证读出的数据是同样的数据类型。
    print('i:',i,'  name:',name)
    if name=='vertex_indices':
        # print(faces_pd[name].values)
        # faces_np[:, i:3]=np.asarray(faces_pd[name], dtype=object)
        # face_V=np.asarray(faces_pd[name].values[:]).reshape(faces_num, 3)
        # print('face_V:',face_V)
        j=0
        while j<faces_num_test:
            faces_np_test[j, 0:3] = faces_pd_test[name].values[j]
            j=j+1
    else:
        # 提取face_colors
        faces_np_test[:, i+2] = faces_pd[name]
print('faces_np_test:',faces_np_test)
'''
2、 mesh预处理：删除黑色面-无label
'''
# i=0
# faces_train=faces_np
# while i<faces_num:
#     if faces_train[i,4]==0 and faces_train[i,5]==0 and faces_train[i,3]==0:
#         print('i ; ',i,'    face_blake:',faces_train[i,:])
#         faces_train=np.delete(faces_train,i,0)
#         print('face_with_colors.shape: ',faces_train.shape)
#         faces_num=faces_num-1
#     else:
#         i=i+1
# print('faces:')
# print(faces_train)
# np.savetxt('E:/MeshSementicSegment/faces_train.txt', faces_train, fmt="%d")
# i=0
# faces_test=faces_np_test
# while i<faces_num_test:
#     if faces_test[i,4]==0 and faces_test[i,5]==0 and faces_test[i,3]==0:
#         print('i ; ',i,'    face_blake:',faces_test[i,:])
#         faces_test=np.delete(faces_test,i,0)
#         print('face_with_colors.shape: ',faces_test.shape)
#         faces_num_test=faces_num_test-1
#     else:
#         i=i+1
# print('faces:')
# print(faces_test)
# np.savetxt('E:/MeshSementicSegment/faces_test.txt', faces_test, fmt="%d")

faces_train = np.loadtxt('E:/MeshSementicSegment/faces_train.txt',dtype=int)  # 读取文件
faces_test = np.loadtxt('E:/MeshSementicSegment/faces_test.txt',dtype=int)  # 读取文件
# print(faces_train)

'''
3、 提取领域mesh
'''
# 确保mesh的顶点和面是numpy array
vertices_train = np.asarray(gt_train_mesh.vertices)
# 确保mesh的顶点和面是numpy array
vertices_test = np.asarray(gt_test_mesh.vertices)
def find_adjacent_faces(faces_train):
    """构建每个面的邻接面列表"""
    # 初始化邻接列表：列是边，行是与边相邻的两个邻面
    adjacency_list = {}
    # 遍历三角面:
    for i, triangle in enumerate(faces_train[:,:3]):
        # print('i: ',i,'triangle:', triangle)  #i:  54943 triangle: [87507 87094 87214]
        # 遍历三角面的边:
        for j in range(3):
            # (j + 1) % 3确保索引是循环的，即当j为2时，(j + 1) % 3将为0。
            # 提取一个面包含的所有边
            edge = (triangle[j], triangle[(j + 1) % 3]) #0,1 1,2 2,0
            # 确保边的顺序:索引排列从小到大
            if edge[1] < edge[0]:
                edge = (edge[1], edge[0])
            # 添加边到邻接列表:
            if edge not in adjacency_list:
                adjacency_list[edge] = []
            # 添加边对应的两个邻面
            adjacency_list[edge].append(i)
    # print('adjacency_list:', adjacency_list)
    #(847515, 847534): [560741, 565099], (846233, 847515): [560741, 562761], (846337, 847628): [560742, 562306],
    # print('adjacency_edge_list.values():',adjacency_list.values())

    face_neighbors = {}
    # 构建每个面的邻接面集合:每个面本身和它的邻面
    for faces in adjacency_list.values():
        # print('faces:',faces)   #[166128, 166197]
        for face in faces:
            if face not in face_neighbors:
                face_neighbors[face] = set()
            face_neighbors[face].update(faces)
    # print('face_neighbors:', face_neighbors)
    #  574391: {574755, 574051, 574541, 574391}, 574055: {574056, 574053, 574341, 574055},

    # # 扩展搜索以满足至少1024个邻接面的要求
    # expanded_neighbors = {}
    # for face in face_neighbors:
    #     # 使用BFS扩展搜索
    #     queue = [face]
    #     visited = set([face])
    #     while len(queue) > 0 and len(visited) < 1024:
    #         current_face = queue.pop(0)
    #         for neighbor in face_neighbors.get(current_face, []):
    #             if neighbor not in visited:
    #                 visited.add(neighbor)
    #                 queue.append(neighbor)
    #
    #     # 更新每个面的邻接面集合
    #     expanded_neighbors[face] = visited
    # print('expanded_neighbors:', expanded_neighbors)

    return face_neighbors
# 构建邻接面列表
face_neighbors = find_adjacent_faces(faces_train)
face_neighbors_test = find_adjacent_faces(faces_test)
def bfs_collect_faces(face_neighbors, start_face, max_faces=2048):
    """使用广度优先搜索来收集紧挨的面"""
    # 初始化visited集合: 用于跟踪已经访问过的面，以防止重复访问。
    visited = set([start_face])
    # 初始化queue队列: deque是双端队列的缩写，它在两端都支持快速的元素插入和删除操作。
    # 将访问queue中的面，同时将这些面的未访问邻接面加入到队列中。
    queue = deque([start_face])
    # 初始化列表，以收集最终的面的索引
    collected_faces = []

    # 当队列不为空且收集的面少于max_faces时，继续搜索
    while queue and len(collected_faces) < max_faces:
        # 从队列中取出当前面的索引：popleft() 移除列表最左端的一个元素，并且返回该元素的值
        current_face = queue.popleft()
        # 将当前面的索引添加到collected_faces中
        collected_faces.append(current_face)
        # 获取当前面的所有邻接面
        for neighbor in face_neighbors[current_face]:
            # 如果邻接面未被访问过，将其添加到队列和visited集合中
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    print('collected_faces:', collected_faces)
    # [70610, 70304, 70609, 70815, 69237, 70816, 70817, 71332,
    # ...
    #  72731, 72838, 72733, 72788, 72825, 73080, 72941]
    return collected_faces

# 随机选择一个三角面
print('labeled_faces_num: ',faces_test.shape[0])   # 580890

# 确定生成submesh个数为m个
m=50
iter=0
while iter <m:
    # face_index =70610
    # face_index = random.randint(0, faces_train.shape[0] - 1)    #70610
    face_index = random.randint(0, faces_test.shape[0] - 1)  # 70610
    # print('random face_index: ',face_index)
    # 从选中的面开始，收集紧挨的1024个面
    collected_faces_indices = bfs_collect_faces(face_neighbors_test, face_index)
    # print('collected_faces_indices.len: ',len(collected_faces_indices)) #1024
    #  collected_faces_indices:[...,4093, 4094, 4095]
    # 提取相邻面
    collected_faces_mesh = np.empty([len(collected_faces_indices),6], dtype = int)
    j=0
    for i in collected_faces_indices:
        collected_faces_mesh[j,:]=faces_train[i,:]
        j=j+1
    # print('collected_faces_mesh.shape: ',collected_faces_mesh.shape)    #(1024, 6)
    # print('collected_faces_mesh: ',collected_faces_mesh)
    #[[ 59  50  11]
     # [ 62  30 101]
     # [ 67   9  86]
     # [ 80  72  27]
     # [ 97  16 111]
     # [ 99  68 102]
     # [100  29  44]
     # [110  20  70]
     # [115  18  58]
     # [117  58  18]
     # [120 110  70]
     # [120  70  88]
    '''
    4、输出submesh数据集
    '''
    def export_faces_and_colors_with_plyfile(gt_train_mesh, collected_faces_mesh,faces_train, file_name):
        '''
        # 获取选定面的顶点索引    就是face的v1 v2 v3
        ''''新顶点映射相关''''
        # # 获取所有独特的顶点索引
        # unique_vertices_indices = np.unique(collected_faces_mesh.flatten())
        # print('unique_vertices_indices: ',unique_vertices_indices)
        # # 创建一个映射，将原始顶点索引映射到新索引
        # # 提取并重映射顶点坐标
        # vertices = np.asarray(gt_train_mesh.vertices)[unique_vertices_indices]
        # vertex_map = {old_idx: new_idx for new_idx, old_idx in enumerate(unique_vertices_indices)}
        # print('vertex_map: ',vertex_map)
        # # 将selected_faces中的每个顶点索引替换为其在vertex_map映射中的新索引
        # # vertex_map.get是从vertex_map字典中获取给定键的值的方法
        # remapped_faces = np.vectorize(vertex_map.get)(selected_faces)
        # print('remapped_faces.shape: ',remapped_faces.shape)    # (1024, 3)
        # [[1092  909  632]
        #  [ 595  898  730]
        #  [ 596  678  317]
        #  ...
        #  [1251 1607 1415]
        #  [1252 1223 1362]
        #  [ 595  730  418]]
    
        # # 创建对应颜色映射
        # # face_colors是与collected_faces_mesh相对应的颜色列表，包含face的vertex_idx和RGB值
        # face_colors_list= []
        # for target_row  in selected_faces:
        #     # print('updated_faces_row:',target_row ) # [1114  994 1125]
        #     target_set = set(target_row )  # 将目标行转换为集合
        #     for row in faces_train:
        #         # print('faces_train_row:', row)  # [856438 855788 856295    128    255    255]
        #         if set(row[:3]) == target_set:  # 比较集合
        #             ''''如果匹配，保存面color，并修改顶点索引''''
        #
        #             face_colors_list.append(row)  # 如果匹配，添加到列表中
        #             # print('face_colors_list:', face_colors_list)
    
        # 将匹配的行列表转换为 numpy 数组
        # face_colors=np.array(face_colors_list)
        # print('face_colors.shape:', face_colors.shape)  #(306, 6)
        # np.savetxt('E:/MeshSementicSegment/face_colors_1.txt', face_colors, fmt="%d")
        # face_ply=np.loadtxt('E:/MeshSementicSegment/face_colors_1.txt',dtype=int)  # 读取文件
        '''
        face_vertex_indices=collected_faces_mesh[:,0:3]
        # print('face_vertex_indices: ',face_vertex_indices)
        face_colors=collected_faces_mesh[:,3:6]
        print('face_colors.shape: ',face_colors.shape)  #(306, 3)
        # 创建顶点列表
        # 原始包含所有顶点的列表
        vertices_xyz = np.array([(x, y, z) for x, y, z in np.asarray(gt_train_mesh.vertices)],
                                dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
        # 仅包含面顶点的列表
        # vertices_xyz = np.array([(x, y, z) for x, y, z in vertices], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
        print('ply_vertices_xyz: ', vertices_xyz)
        print('ply_vertices_xyz.shape: ', vertices_xyz.shape)
        #[(-13.6626, -0.545707,  65.076 ) (-13.581 ,  1.93622 , -43.2188)
        # (-13.5129, -1.93234 ,  65.3708) ... (-12.8723, -2.08013 , -44.6719)
        # (-12.8719,  0.733936, -43.9954) (-12.8253,  1.95508 , -43.5398)]
        # 创建面列表;'u1'- 无符号8位整型
        n = len(face_vertex_indices)  # 假设face_vertex_indices和face_colors的长度相同
        # 创建一个空的结构化数组
        faces = np.empty(n, dtype=[('vertex_indices', 'i4', (3,)), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
        # 填充数据
        for i, (face, color) in enumerate(zip(face_vertex_indices, face_colors)):
            faces['vertex_indices'][i] = face
            faces['red'][i], faces['green'][i], faces['blue'][i] = color
        print('ply_faces: ',faces)

        # 创建PlyElement对象
        vertex_element = PlyElement.describe(vertices_xyz, 'vertex')
        face_element = PlyElement.describe(faces, 'face')

        # 写入PLY文件
        PlyData([vertex_element, face_element], text=True).write(file_name)
    # iter_name=r'E:/MeshSementicSegment/train/sub_mesh_'+str(iter)+'.ply'
    '''train
    sub_mesh_=export_faces_and_colors_with_plyfile(gt_train_mesh, collected_faces_mesh,faces_train, r'E:/MeshSementicSegment/train/sub_mesh_'+str(iter+1)+'.ply')
    # open3d读写会遗失所有face_color信息
    mesh_=o3d.io.read_triangle_mesh(r'E:/MeshSementicSegment/train/sub_mesh_'+str(iter+1)+'.ply')
    mesh_o3d=mesh_.remove_unreferenced_vertices()
    sub_mesh_o3d=o3d.io.write_triangle_mesh(r'E:/MeshSementicSegment/train/sub_mesh_'+str(iter+1)+'_o3d.ply',mesh_o3d,write_ascii=True)
    # trimesh
    mesh_trimesh =  trimesh.load_mesh(r'E:/MeshSementicSegment/train/sub_mesh_' + str(iter+1) + '.ply')
    # mesh.Trimesh.remove_unreferenced_vertices()   #在读入时会自动进行
    # 强制导出二进制格式的ply文件，文本打开会乱码，但读数据不影响
    mesh_trimesh.export(r'E:/MeshSementicSegment/train/sub_mesh_' + str(iter+1) + '_trimesh.ply' )
    '''
    # test
    sub_mesh_=export_faces_and_colors_with_plyfile(gt_test_mesh, collected_faces_mesh,faces_test, r'D:\Sematic_Dataset\RueMonge2014-varcity3dchallenge\data\ruemonge428\test/sub_mesh_'+str(iter+1)+'.ply')
    # open3d读写会遗失所有face_color信息
    mesh_=o3d.io.read_triangle_mesh(r'D:\Sematic_Dataset\RueMonge2014-varcity3dchallenge\data\ruemonge428\test/sub_mesh_'+str(iter+1)+'.ply')
    mesh_o3d=mesh_.remove_unreferenced_vertices()
    sub_mesh_o3d=o3d.io.write_triangle_mesh(r'D:\Sematic_Dataset\RueMonge2014-varcity3dchallenge\data\ruemonge428\test/sub_mesh_'+str(iter+1)+'_o3d.ply',mesh_o3d,write_ascii=True)
    # trimesh
    mesh_trimesh =  trimesh.load_mesh(r'D:\Sematic_Dataset\RueMonge2014-varcity3dchallenge\data\ruemonge428\test/sub_mesh_' + str(iter+1) + '.ply')
    # mesh.Trimesh.remove_unreferenced_vertices()   #在读入时会自动进行
    # 强制导出二进制格式的ply文件，文本打开会乱码，但读数据不影响
    mesh_trimesh.export(r'D:\Sematic_Dataset\RueMonge2014-varcity3dchallenge\data\ruemonge428\test/sub_mesh_' + str(iter+1) + '_trimesh.ply' )
    iter=iter+1
