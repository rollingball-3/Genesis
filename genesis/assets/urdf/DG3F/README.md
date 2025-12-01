# DG3F Gripper URDF Model

这是一个独立的DG3F三指夹爪URDF模型包。

## 文件结构

```
DG3F_model/
├── meshes/                      # STL mesh文件
│   ├── delto_base_link.stl     # 基座连杆
│   ├── link_01.stl             # 手指连杆1
│   ├── link_02.stl             # 手指连杆2
│   ├── link_03.stl             # 手指连杆3
│   ├── link_04.stl             # 手指连杆4
│   └── link_tip_high.stl       # 手指尖端
├── urdf/                        # URDF文件
│   └── delto_gripper_3f.urdf   # 主URDF描述文件
└── README.md                    # 说明文档
```

## 模型特性

- **三指夹爪**: 3个独立控制的手指 (F1, F2, F3)
- **每指4个关节**: F1M1-F1M4, F2M1-F2M4, F3M1-F3M4
- **固定指尖**: 每个手指的尖端通过固定关节连接
- **相对路径**: mesh文件使用相对路径引用，便于移动和部署

## 使用方法

### 1. 作为独立模型使用

将整个`DG3F_model`文件夹复制到您的项目中，URDF文件使用相对路径引用mesh文件，可直接加载。

### 2. 集成到ROS2包中

将URDF文件复制到您的ROS2包的`urdf/`目录，将mesh文件复制到`meshes/`目录，并根据需要修改mesh路径为ROS2包路径格式：

```xml
<mesh filename="package://your_package_name/meshes/link_01.stl"/>
```

## 关节限位

各关节的运动范围：

- **F1M1**: -1.0472 ~ 1.0472 rad (-60° ~ 60°)
- **F1M2/F2M2/F3M2**: -1.76278 ~ 1.76278 rad (-101° ~ 101°)
- **F1M3/F2M3/F3M3**: -0.15708 ~ 2.53073 rad (-9° ~ 145°)
- **F1M4/F2M4/F3M4**: -0.226893 ~ 2.02458 rad (-13° ~ 116°)
- **F2M1**: -1.91986 ~ 0.139626 rad (-110° ~ 8°)
- **F3M1**: -0.0872665 ~ 2.00713 rad (-5° ~ 115°)

## 版权信息

Licensed under the BSD-3-Clause
Copyright (c) 2025 Tesollo Inc.
Reference: https://tesollo.com/dg-3f/
