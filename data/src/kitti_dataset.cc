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

#include <sstream>

#include "utils.hh"

#include "kitti_dataset.hh"

KittiDataset::KittiDataset(bfs::path basePath, Mode mode)
{
    switch (mode) {
    case Mode::Train:
        m_groundTruthPath = basePath / bfs::path("training") / bfs::path("label_2");
        m_leftImgPath = basePath / bfs::path("training") / bfs::path("image_2");
        m_extractBoundingboxes = true;
        break;
    case Mode::Test:
        m_groundTruthPath = basePath / bfs::path("testing") / bfs::path("label_2");
        m_leftImgPath = basePath / bfs::path("testing") / bfs::path("image_2");
        m_extractBoundingboxes = false;
        break;
    case Mode::Val:
        m_groundTruthPath = basePath / bfs::path("validation") / bfs::path("label_2");
        m_leftImgPath = basePath / bfs::path("validation") / bfs::path("image_2");
        m_extractBoundingboxes = true;
        break;
    default:
        CHECK(false, "Unknown mode!");
    }

    for (auto &entry : bfs::recursive_directory_iterator(m_groundTruthPath)) {
        if (entry.path().extension() == ".txt") {
            auto relativePath = bfs::relative(entry.path(), m_groundTruthPath);
            std::string key = relativePath.string();
            key = key.substr(0, key.length() - entry.path().extension().string().length());
            m_keys.push_back(key);
        }
    }
    std::sort(m_keys.begin(), m_keys.end());
}

std::shared_ptr<DatasetEntry> KittiDataset::get(std::size_t i)
{
    CHECK(i < m_keys.size(), "Index out of range");
    auto key = m_keys[i];
    auto result = std::make_shared<DatasetEntry>();
    auto leftImgPath = m_leftImgPath / bfs::path(key + ".png");
    cv::Mat leftImg = cv::imread(leftImgPath.string());
    CHECK(leftImg.data, "Failed to read image " + leftImgPath.string());
    result->input.left = toFloatMat(leftImg);
    if (m_extractBoundingboxes) {
        auto gtPath = m_groundTruthPath / bfs::path(key + ".txt");
        std::ifstream gtFs(gtPath.string());
        auto [bbDontCareAreas, bbList] = parseGt(gtFs, result->input.left.size());
        result->gt.bbDontCareAreas = bbDontCareAreas;
        result->gt.bbList = bbList;
    }
    result->metadata.originalWidth = result->input.left.cols;
    result->metadata.originalHeight = result->input.left.rows;
    result->metadata.canFlip = true;
    result->metadata.horizontalFov = 90.0;
    result->metadata.key = key;
    return result;
}

std::tuple<cv::Mat, BoundingBoxList> KittiDataset::parseGt(std::ifstream &gtFs, cv::Size imageSize)
{
    cv::Mat bbDontCareImg(imageSize, CV_32SC1, cv::Scalar(m_boundingBoxValidLabel));
    BoundingBoxList bbList;
    bbList.valid = true;
    bbList.width = imageSize.width;
    bbList.height = imageSize.height;

    std::string line;
    std::stringstream splitter;
    int64_t objectId = getRandomId();
    while (std::getline(gtFs, line)) {
        splitter << line;
        std::string cls;
        splitter >> cls;
        double ddummy;
        int idummy;
        splitter >> ddummy;
        splitter >> idummy;
        splitter >> ddummy;
        double x1, y1, x2, y2;
        splitter >> x1 >> y1 >> x2 >> y2;
        splitter.clear();
        if (m_instanceDict.count(cls) == 0) {
            /* Draw don't care image for bounding boxes. */
            cv::rectangle(bbDontCareImg, cv::Rect(x1, y1, x2 - x1, y2 - y1),
                          cv::Scalar(m_boundingBoxDontCareLabel), -1);
        } else {
            /* Generate bounding box list */
            BoundingBox boundingBox;
            boundingBox.id = objectId++;
            boundingBox.cls = m_instanceDict.at(cls);
            boundingBox.x1 = x1;
            boundingBox.x2 = x2;
            boundingBox.y1 = y1;
            boundingBox.y2 = y2;
            bbList.boxes.push_back(boundingBox);
        }
    }

    return {bbDontCareImg, bbList};
}
