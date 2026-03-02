#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
XYZ 坐标轴可视化节点
用于在 RViz 中显示带刻度的坐标轴，方便调参
"""

import rospy
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA

class AxisVisualizer:
    def __init__(self):
        rospy.init_node('axis_visualizer', anonymous=True)
        
        # 参数 - 与 rqt_reconfigure 中的滤波范围匹配
        self.frame_id = rospy.get_param('~frame_id', 'livox_frame')
        
        # X轴范围 (前方)
        self.x_min = rospy.get_param('~x_min', 0.0)
        self.x_max = rospy.get_param('~x_max', 35.0)
        
        # Y轴范围 (左右)
        self.y_min = rospy.get_param('~y_min', -10.0)
        self.y_max = rospy.get_param('~y_max', 8.0)
        
        # Z轴范围 (上下)
        self.z_min = rospy.get_param('~z_min', -50.0)
        self.z_max = rospy.get_param('~z_max', 7.0)
        
        self.tick_interval = rospy.get_param('~tick_interval', 5.0)  # 刻度间隔
        self.axis_width = rospy.get_param('~axis_width', 0.1)  # 轴线宽度
        
        # 发布者
        self.marker_pub = rospy.Publisher('/axis_markers', MarkerArray, queue_size=1, latch=True)
        
        # 发布坐标轴
        self.publish_axis_markers()
        
        rospy.loginfo("Axis visualizer started. Frame: %s", self.frame_id)
        rospy.loginfo("  X: %.1f to %.1f, Y: %.1f to %.1f, Z: %.1f to %.1f", 
                      self.x_min, self.x_max, self.y_min, self.y_max, self.z_min, self.z_max)
    
    def create_arrow_marker(self, marker_id, start, end, color, scale=0.1):
        """创建箭头标记"""
        marker = Marker()
        marker.header.frame_id = self.frame_id
        marker.header.stamp = rospy.Time.now()
        marker.ns = "axis_arrows"
        marker.id = marker_id
        marker.type = Marker.ARROW
        marker.action = Marker.ADD
        
        # 起点和终点
        marker.points.append(Point(start[0], start[1], start[2]))
        marker.points.append(Point(end[0], end[1], end[2]))
        
        # 箭头尺寸: shaft diameter, head diameter, head length
        marker.scale.x = scale * 0.5  # 轴直径
        marker.scale.y = scale * 1.0  # 箭头直径
        marker.scale.z = scale * 1.5  # 箭头长度
        
        marker.color = color
        marker.lifetime = rospy.Duration(0)
        
        return marker
    
    def create_text_marker(self, marker_id, position, text, color, scale=1.0):
        """创建文字标记"""
        marker = Marker()
        marker.header.frame_id = self.frame_id
        marker.header.stamp = rospy.Time.now()
        marker.ns = "axis_labels"
        marker.id = marker_id
        marker.type = Marker.TEXT_VIEW_FACING
        marker.action = Marker.ADD
        
        marker.pose.position.x = position[0]
        marker.pose.position.y = position[1]
        marker.pose.position.z = position[2]
        marker.pose.orientation.w = 1.0
        
        marker.scale.z = scale
        marker.color = color
        marker.text = text
        marker.lifetime = rospy.Duration(0)
        
        return marker
    
    def create_tick_marker(self, marker_id, position, axis, color, length=0.3):
        """创建刻度线标记"""
        marker = Marker()
        marker.header.frame_id = self.frame_id
        marker.header.stamp = rospy.Time.now()
        marker.ns = "axis_ticks"
        marker.id = marker_id
        marker.type = Marker.LINE_LIST
        marker.action = Marker.ADD
        
        # 根据轴方向创建垂直的刻度线
        if axis == 'x':
            marker.points.append(Point(position, -length, 0))
            marker.points.append(Point(position, length, 0))
            marker.points.append(Point(position, 0, -length))
            marker.points.append(Point(position, 0, length))
        elif axis == 'y':
            marker.points.append(Point(-length, position, 0))
            marker.points.append(Point(length, position, 0))
            marker.points.append(Point(0, position, -length))
            marker.points.append(Point(0, position, length))
        elif axis == 'z':
            marker.points.append(Point(-length, 0, position))
            marker.points.append(Point(length, 0, position))
            marker.points.append(Point(0, -length, position))
            marker.points.append(Point(0, length, position))
        
        marker.scale.x = 0.05
        marker.color = color
        marker.lifetime = rospy.Duration(0)
        
        return marker
    
    def create_plane_grid(self, marker_id, plane, size, interval, color):
        """创建平面网格"""
        marker = Marker()
        marker.header.frame_id = self.frame_id
        marker.header.stamp = rospy.Time.now()
        marker.ns = "axis_grid"
        marker.id = marker_id
        marker.type = Marker.LINE_LIST
        marker.action = Marker.ADD
        
        half_size = size / 2.0
        
        # 根据平面创建网格线
        i = -half_size
        while i <= half_size:
            if plane == 'xy':
                # 平行于 X 轴的线
                marker.points.append(Point(-half_size, i, 0))
                marker.points.append(Point(half_size, i, 0))
                # 平行于 Y 轴的线
                marker.points.append(Point(i, -half_size, 0))
                marker.points.append(Point(i, half_size, 0))
            elif plane == 'xz':
                marker.points.append(Point(-half_size, 0, i))
                marker.points.append(Point(half_size, 0, i))
                marker.points.append(Point(i, 0, -half_size))
                marker.points.append(Point(i, 0, half_size))
            elif plane == 'yz':
                marker.points.append(Point(0, -half_size, i))
                marker.points.append(Point(0, half_size, i))
                marker.points.append(Point(0, i, -half_size))
                marker.points.append(Point(0, i, half_size))
            i += interval
        
        marker.scale.x = 0.02
        marker.color = color
        marker.lifetime = rospy.Duration(0)
        
        return marker
    
    def publish_axis_markers(self):
        """发布所有坐标轴标记"""
        markers = MarkerArray()
        marker_id = 0
        
        # 颜色定义 (RGBA)
        red = ColorRGBA(1.0, 0.0, 0.0, 1.0)      # X轴 - 红色
        green = ColorRGBA(0.0, 1.0, 0.0, 1.0)    # Y轴 - 绿色
        blue = ColorRGBA(0.0, 0.5, 1.0, 1.0)     # Z轴 - 蓝色
        white = ColorRGBA(1.0, 1.0, 1.0, 0.8)    # 文字
        yellow = ColorRGBA(1.0, 1.0, 0.0, 0.5)   # 边界线
        
        # ===== X轴 (红色) - 从 x_min 到 x_max =====
        markers.markers.append(self.create_arrow_marker(
            marker_id, [self.x_min, 0, 0], [self.x_max, 0, 0], red, self.axis_width))
        marker_id += 1
        
        # X轴标签
        markers.markers.append(self.create_text_marker(
            marker_id, [self.x_max + 1, 0, 0], "X (前) max=%.0f" % self.x_max, red, 1.2))
        marker_id += 1
        markers.markers.append(self.create_text_marker(
            marker_id, [self.x_min - 1, 0, 0], "min=%.0f" % self.x_min, red, 1.0))
        marker_id += 1
        
        # X轴刻度和数字
        x = self.tick_interval * int(self.x_min / self.tick_interval)
        while x <= self.x_max:
            if x >= self.x_min:
                markers.markers.append(self.create_tick_marker(marker_id, x, 'x', red, 0.2))
                marker_id += 1
                markers.markers.append(self.create_text_marker(
                    marker_id, [x, -0.8, 0], "%.0f" % x, white, 0.8))
                marker_id += 1
            x += self.tick_interval
        
        # ===== Y轴 (绿色) - 从 y_min 到 y_max =====
        markers.markers.append(self.create_arrow_marker(
            marker_id, [0, self.y_min, 0], [0, self.y_max, 0], green, self.axis_width))
        marker_id += 1
        
        # Y轴标签
        markers.markers.append(self.create_text_marker(
            marker_id, [0, self.y_max + 1, 0], "Y (左) max=%.0f" % self.y_max, green, 1.2))
        marker_id += 1
        markers.markers.append(self.create_text_marker(
            marker_id, [0, self.y_min - 1, 0], "Y (右) min=%.0f" % self.y_min, green, 1.2))
        marker_id += 1
        
        # Y轴刻度
        y = self.tick_interval * int(self.y_min / self.tick_interval)
        while y <= self.y_max:
            if y >= self.y_min and y != 0:
                markers.markers.append(self.create_tick_marker(marker_id, y, 'y', green, 0.2))
                marker_id += 1
                markers.markers.append(self.create_text_marker(
                    marker_id, [-0.8, y, 0], "%.0f" % y, white, 0.8))
                marker_id += 1
            y += self.tick_interval
        
        # ===== Z轴 (蓝色) - 从 z_min 到 z_max =====
        markers.markers.append(self.create_arrow_marker(
            marker_id, [0, 0, self.z_min], [0, 0, self.z_max], blue, self.axis_width))
        marker_id += 1
        
        # Z轴标签
        markers.markers.append(self.create_text_marker(
            marker_id, [0, 0, self.z_max + 1], "Z (上) max=%.0f" % self.z_max, blue, 1.2))
        marker_id += 1
        markers.markers.append(self.create_text_marker(
            marker_id, [0, 0, self.z_min - 1], "Z (下) min=%.0f" % self.z_min, blue, 1.2))
        marker_id += 1
        
        # Z轴刻度 (每10米一个刻度，因为Z范围很大)
        z_tick = 10.0 if (self.z_max - self.z_min) > 30 else self.tick_interval
        z = z_tick * int(self.z_min / z_tick)
        while z <= self.z_max:
            if z >= self.z_min and z != 0:
                markers.markers.append(self.create_tick_marker(marker_id, z, 'z', blue, 0.2))
                marker_id += 1
                markers.markers.append(self.create_text_marker(
                    marker_id, [-0.8, 0, z], "%.0f" % z, white, 0.8))
                marker_id += 1
            z += z_tick
        
        # ===== 原点标记 =====
        markers.markers.append(self.create_text_marker(
            marker_id, [-1.5, -1.5, 0.5], "O (原点)", white, 1.2))
        marker_id += 1
        
        # ===== 边界框线 (显示滤波区域) =====
        markers.markers.append(self.create_bounding_box(marker_id, yellow))
        marker_id += 1
        
        # 发布
        self.marker_pub.publish(markers)
        rospy.loginfo("Published %d axis markers", len(markers.markers))
    
    def create_bounding_box(self, marker_id, color):
        """创建滤波区域边界框"""
        marker = Marker()
        marker.header.frame_id = self.frame_id
        marker.header.stamp = rospy.Time.now()
        marker.ns = "filter_bounds"
        marker.id = marker_id
        marker.type = Marker.LINE_LIST
        marker.action = Marker.ADD
        
        # 8个角点
        x0, x1 = self.x_min, self.x_max
        y0, y1 = self.y_min, self.y_max
        z0, z1 = self.z_min, self.z_max
        
        # 底面 4条边
        marker.points.append(Point(x0, y0, z0)); marker.points.append(Point(x1, y0, z0))
        marker.points.append(Point(x1, y0, z0)); marker.points.append(Point(x1, y1, z0))
        marker.points.append(Point(x1, y1, z0)); marker.points.append(Point(x0, y1, z0))
        marker.points.append(Point(x0, y1, z0)); marker.points.append(Point(x0, y0, z0))
        
        # 顶面 4条边
        marker.points.append(Point(x0, y0, z1)); marker.points.append(Point(x1, y0, z1))
        marker.points.append(Point(x1, y0, z1)); marker.points.append(Point(x1, y1, z1))
        marker.points.append(Point(x1, y1, z1)); marker.points.append(Point(x0, y1, z1))
        marker.points.append(Point(x0, y1, z1)); marker.points.append(Point(x0, y0, z1))
        
        # 垂直 4条边
        marker.points.append(Point(x0, y0, z0)); marker.points.append(Point(x0, y0, z1))
        marker.points.append(Point(x1, y0, z0)); marker.points.append(Point(x1, y0, z1))
        marker.points.append(Point(x1, y1, z0)); marker.points.append(Point(x1, y1, z1))
        marker.points.append(Point(x0, y1, z0)); marker.points.append(Point(x0, y1, z1))
        
        marker.scale.x = 0.05
        marker.color = color
        marker.lifetime = rospy.Duration(0)
        
        return marker
    
    def run(self):
        """运行节点"""
        rate = rospy.Rate(0.5)  # 每2秒更新一次
        while not rospy.is_shutdown():
            self.publish_axis_markers()
            rate.sleep()

if __name__ == '__main__':
    try:
        visualizer = AxisVisualizer()
        visualizer.run()
    except rospy.ROSInterruptException:
        pass
