/**************************************************************************
 * NNAD (Neural Networks for Automated Driving) training scripts          *
 * Copyright (C) 2019 FZI Research Center for Information Technology      *
 *                                                                        *
 * This program is free software: you can redistribute it and/or modify   *
 * it under the terms of the GNU General Public License as published by   *
 * the Free Software Foundation, either version 3 of the License, or      *
 * (at your option) any later version.                                    *
 *                                                                        *
 * This program is distributed in the hope that it will be useful,        *
 * but WITHOUT ANY WARRANTY; without even the implied warranty of         *
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the          *
 * GNU General Public License for more details.                           *
 *                                                                        *
 * You should have received a copy of the GNU General Public License      *
 * along with this program.  If not, see <https://www.gnu.org/licenses/>. *
 **************************************************************************/

#include <iostream>
#include <fstream>
#include <cstring>
#include <sstream>
#include <iomanip>
#include <random>
#include <chrono>

#include "utils.hh"

#include "sintel_dataset.hh"

SintelDataset::SintelDataset(bfs::path basePath)
{
    auto seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    m_generator = std::mt19937(seed);

    m_flowPath = basePath / bfs::path("training") / bfs::path("flow");
    m_imagePath = basePath / bfs::path("training") / bfs::path("final");

    for (auto &entry : bfs::recursive_directory_iterator(m_flowPath)) {
        if (entry.path().extension() == ".flo") {
            auto relativePath = bfs::relative(entry.path(), m_flowPath);
            std::string key = relativePath.string();
            key = key.substr(0, key.length() - std::string(".flo").length());
            m_keys.push_back(key);
        }
    }
    std::sort(m_keys.begin(), m_keys.end());
}

std::string SintelDataset::nextKey(std::string &key) const
{
    std::string remainder = key;
    auto pos = remainder.rfind("_");
    std::string prefix = remainder.substr(0, pos);
    std::string frameno = remainder.substr(pos + 1);

    int numericFrameno = std::stoi(frameno) + 1;

    std::ostringstream result;
    result << prefix << "_" << std::setw(4) << std::setfill('0') << numericFrameno;

    return result.str();
}

std::shared_ptr<DatasetEntry> SintelDataset::get(std::size_t i)
{
    CHECK(i < m_keys.size(), "Index out of range");
    auto key = m_keys[i];
    auto result = std::make_shared<DatasetEntry>();

    /* We crop a random roi here to make sure that the image height is a multiple of 128.
     * This should really be done in the augmentation / preparation code but for now it is easier here. */
    cv::Rect roi(0, m_yDistribution(m_generator), 1024, 384);

    auto currentImgPath = m_imagePath / bfs::path(nextKey(key) + std::string(".png"));
    cv::Mat currentImg = cv::imread(currentImgPath.string());
    CHECK(currentImg.data, "Failed to read image " + currentImgPath.string());
    result->input.left = toFloatMat(currentImg);
    result->input.left = result->input.left(roi);

    auto prevImgPath = m_imagePath / bfs::path(key + std::string(".png"));
    cv::Mat prevImg = cv::imread(prevImgPath.string());
    CHECK(prevImg.data, "Failed to read image " + prevImgPath.string());
    result->input.prevLeft = toFloatMat(prevImg);
    result->input.prevLeft = result->input.prevLeft(roi);

    auto flowPath = m_flowPath / bfs::path(key + std::string(".flo"));
    cv::Mat flowImg = readFlowImg(flowPath.string());
    flowImg = flowImg(roi);
    result->gt.flow = flowImg;

    result->metadata.originalWidth = result->input.left.cols;
    result->metadata.originalHeight = result->input.left.rows;
    result->metadata.canFlip = false;
    result->metadata.horizontalFov = -1.0;
    result->metadata.key = key;
    return result;
}

cv::Mat SintelDataset::readFlowImg(std::string filename) const
{
    std::ifstream file(filename, std::ios::in | std::ios::binary);
    CHECK(file.is_open(), "Failed to open file " + filename);
    char header[4];
    file.read(header, 4);
    CHECK(memcmp(header, "PIEH", 4) == 0, "Header does not match!");
    int32_t width, height;
    file.read(reinterpret_cast<char *>(&width), sizeof(int32_t));
    file.read(reinterpret_cast<char *>(&height), sizeof(int32_t));
    cv::Mat result(height, width, CV_32FC2);
    file.read(reinterpret_cast<char *>(result.data), 2 * width * height * sizeof(float));
    file.close();

    /* Change the channel order so that it matches what is expected by tfa.image.dense_image_warp */
    cv::Mat swappedResult(result.rows, result.cols, CV_32FC2);
    int mapping[] = {0, 1, 1, 0};
    cv::mixChannels(&result, 1, &swappedResult, 1, mapping, 2);
    return swappedResult;
}
