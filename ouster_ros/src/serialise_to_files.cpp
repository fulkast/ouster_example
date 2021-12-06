//
// Created by fu on 01/12/2021.
//

#include <pcl/conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/image_encodings.h>

#include <opencv2/imgcodecs.hpp>

#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "ouster/client.h"
#include "ouster/image_processing.h"
#include "ouster/types.h"
#include "ouster_ros/OSConfigSrv.h"
#include "ouster_ros/ros.h"

namespace sensor = ouster::sensor;
namespace viz = ouster::viz;

using pixel_type = uint16_t;
using signed_pixel_type = float;
const size_t pixel_value_max = std::numeric_limits<pixel_type>::max();
const auto signed_pixel_value_max = std::numeric_limits<signed_pixel_type>::max();

sensor_msgs::ImagePtr make_image_msg(size_t H, size_t W,
                                     const ros::Time& stamp) {
    sensor_msgs::ImagePtr msg{new sensor_msgs::Image{}};
    msg->width = W;
    msg->height = H;
    msg->step = W * sizeof(pixel_type);
    msg->encoding = sensor_msgs::image_encodings::MONO16;
    msg->data.resize(W * H * sizeof(pixel_type));
    msg->header.stamp = stamp;

    return msg;
}

sensor_msgs::ImagePtr make_fp_image_msg(size_t H, size_t W, const ros::Time& stamp) {
    sensor_msgs::ImagePtr msg {new sensor_msgs::Image{}};
    msg->width = W;
    msg->height = H;
    msg->step = W * sizeof(signed_pixel_type);
    msg->encoding = sensor_msgs::image_encodings::TYPE_32FC1;
    msg->data.resize(W * H * sizeof(signed_pixel_type));
    msg->header.stamp = stamp;
    return msg;
}

cv_bridge::CvImagePtr prev_ir_cv_ptr;

int main(int argc, char** argv) {
    ros::init(argc, argv, "serialise_to_files_node");
    ros::NodeHandle nh("~");

    ouster_ros::OSConfigSrv cfg{};
    auto client = nh.serviceClient<ouster_ros::OSConfigSrv>("os_config");
    client.waitForExistence();
    if (!client.call(cfg)) {
        ROS_ERROR("Calling os config service failed");
        return EXIT_FAILURE;
    }

    auto info = sensor::parse_metadata(cfg.response.metadata);
    size_t H = info.format.pixels_per_column;
    size_t W = info.format.columns_per_frame;

    const auto& px_offset = info.format.pixel_shift_by_row;

    ros::Publisher range_image_pub =
        nh.advertise<sensor_msgs::Image>("range_image", 100);
    ros::Publisher nearir_image_pub =
        nh.advertise<sensor_msgs::Image>("nearir_image", 100);
    ros::Publisher signal_image_pub =
        nh.advertise<sensor_msgs::Image>("signal_image", 100);
    ros::Publisher reflec_image_pub =
        nh.advertise<sensor_msgs::Image>("reflec_image", 100);

    ouster_ros::Cloud cloud{};

    viz::AutoExposure nearir_ae, signal_ae, reflec_ae;
    viz::BeamUniformityCorrector nearir_buc;

    ouster::img_t<double> nearir_image_eigen(H, W);
    ouster::img_t<double> signal_image_eigen(H, W);
    ouster::img_t<double> reflec_image_eigen(H, W);
    ouster::img_t<double> x_image_eigen(H, W);
    ouster::img_t<double> y_image_eigen(H, W);
    ouster::img_t<double> z_image_eigen(H, W);

    auto cloud_handler = [&](const sensor_msgs::PointCloud2::ConstPtr& m) {
        pcl::fromROSMsg(*m, cloud);

        auto range_image = make_image_msg(H, W, m->header.stamp);
        auto nearir_image = make_image_msg(H, W, m->header.stamp);
        auto signal_image = make_image_msg(H, W, m->header.stamp);
        auto reflec_image = make_image_msg(H, W, m->header.stamp);
        auto x_image = make_fp_image_msg(H, W, m->header.stamp);
        auto y_image = make_fp_image_msg(H, W, m->header.stamp);
        auto z_image = make_fp_image_msg(H, W, m->header.stamp);

        // views into message data
        auto range_image_map = Eigen::Map<ouster::img_t<pixel_type>>(
            (pixel_type*)range_image->data.data(), H, W);
        auto nearir_image_map = Eigen::Map<ouster::img_t<pixel_type>>(
            (pixel_type*)nearir_image->data.data(), H, W);
        auto signal_image_map = Eigen::Map<ouster::img_t<pixel_type>>(
            (pixel_type*)signal_image->data.data(), H, W);
        auto reflec_image_map = Eigen::Map<ouster::img_t<pixel_type>>(
            (pixel_type*)reflec_image->data.data(), H, W);
        auto x_image_map = Eigen::Map<ouster::img_t<signed_pixel_type>>(
            (signed_pixel_type*)x_image->data.data(), H, W);
        auto y_image_map = Eigen::Map<ouster::img_t<signed_pixel_type>>(
            (signed_pixel_type*)y_image->data.data(), H, W);
        auto z_image_map = Eigen::Map<ouster::img_t<signed_pixel_type>>(
            (signed_pixel_type*)z_image->data.data(), H, W);


        // copy data out of Cloud message, with destaggering
        for (size_t u = 0; u < H; u++) {
            for (size_t v = 0; v < W; v++) {
                const size_t vv = (v + W - px_offset[u]) % W;
                const auto& pt = cloud[u * W + vv];

                // 16 bit img: use 4mm resolution and throw out returns > 260m
                auto r = (pt.range + 0b10) >> 2;
                range_image_map(u, v) = r > pixel_value_max ? 0 : r;

                nearir_image_eigen(u, v) = pt.ambient;
                signal_image_eigen(u, v) = pt.intensity;
                reflec_image_eigen(u, v) = pt.reflectivity;
                x_image_eigen(u,v) = pt.x;
                y_image_eigen(u,v) = pt.y;
                z_image_eigen(u,v) = pt.z;
            }
        }

        // image processing
        nearir_buc(nearir_image_eigen);
        nearir_ae(nearir_image_eigen);
        signal_ae(signal_image_eigen);
        reflec_ae(reflec_image_eigen);
        nearir_image_eigen = nearir_image_eigen.sqrt();
        signal_image_eigen = signal_image_eigen.sqrt();

        // copy data into image messages
        nearir_image_map =
            (nearir_image_eigen * pixel_value_max).cast<pixel_type>();
        signal_image_map =
            (signal_image_eigen * pixel_value_max).cast<pixel_type>();
        reflec_image_map =
            (reflec_image_eigen * pixel_value_max).cast<pixel_type>();
        x_image_map =
            (x_image_eigen).cast<signed_pixel_type>();
        y_image_map =
            (y_image_eigen).cast<signed_pixel_type>();
        z_image_map =
            (z_image_eigen).cast<signed_pixel_type>();

        if (prev_ir_cv_ptr){
            cv::imwrite("/tmp/previous_ir.tif", prev_ir_cv_ptr->image);
        }

        const auto nearir_cv_ptr =
            cv_bridge::toCvShare(nearir_image, sensor_msgs::image_encodings::MONO16);
        prev_ir_cv_ptr =
            cv_bridge::toCvCopy(nearir_image, sensor_msgs::image_encodings::MONO16);
        const auto range_cv_ptr =
            cv_bridge::toCvShare(range_image, sensor_msgs::image_encodings::MONO16);
        const auto x_cv_ptr =
            cv_bridge::toCvShare(x_image, sensor_msgs::image_encodings::TYPE_32FC1);
        const auto y_cv_ptr =
            cv_bridge::toCvShare(y_image, sensor_msgs::image_encodings::TYPE_32FC1);
        const auto z_cv_ptr =
            cv_bridge::toCvShare(z_image, sensor_msgs::image_encodings::TYPE_32FC1);

        cv::imwrite("/tmp/latest_ir.tif", nearir_cv_ptr->image);
        cv::imwrite("/tmp/latest_range.tif", range_cv_ptr->image);
        cv::imwrite("/tmp/latest_x.tif", x_cv_ptr->image);
        cv::imwrite("/tmp/latest_y.tif", y_cv_ptr->image);
        cv::imwrite("/tmp/latest_z.tif", z_cv_ptr->image);

        // publish
        range_image_pub.publish(range_image);
        nearir_image_pub.publish(nearir_image);
        signal_image_pub.publish(signal_image);
        reflec_image_pub.publish(reflec_image);
    };

    auto pc_sub =
        nh.subscribe<sensor_msgs::PointCloud2>("points", 100, cloud_handler);

    ros::spin();
    return EXIT_SUCCESS;
}
