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
#include <iomanip>

#include "utils.hh"

#include "cityscapes_dataset.hh"

CityscapesDataset::CityscapesDataset(bfs::path basePath, Mode mode)
{
    switch (mode) {
    case Mode::Train:
        m_groundTruthPath = basePath / bfs::path("gtFine") / bfs::path("train");
        m_leftImgPath = basePath / bfs::path("leftImg8bit") / bfs::path("train");
        m_prevLeftImgPath = basePath / bfs::path("leftImg8bit_sequence") / bfs::path("train");
        m_groundTruthSubstring = std::string("_gtFine_polygons.json");
        m_leftImgSubstring = std::string("_leftImg8bit.png");
        m_extractBoundingboxes = true;
        m_hasSequence = true;
        m_fov = 50.0;
        break;
    case Mode::TrainExtra:
        m_groundTruthPath = basePath / bfs::path("gtCoarse") / bfs::path("train_extra");
        m_leftImgPath = basePath / bfs::path("leftImg8bit") / bfs::path("train_extra");
        m_prevLeftImgPath = basePath / bfs::path("leftImg8bit") / bfs::path("train_extra");
        m_groundTruthSubstring = std::string("_gtCoarse_polygons.json");
        m_leftImgSubstring = std::string("_leftImg8bit.png");
        m_extractBoundingboxes = false;
        m_hasSequence = false;
        m_fov = 50.0;
        break;
    case Mode::TrainBertha:
        m_groundTruthPath = basePath / bfs::path("gtFine") / bfs::path("train_bertha");
        m_leftImgPath = basePath / bfs::path("leftImg8bit") / bfs::path("train_bertha");
        m_prevLeftImgPath = basePath / bfs::path("leftImg8bit") / bfs::path("train_bertha");
        m_groundTruthSubstring = std::string("_gtFine_polygons.json");
        m_leftImgSubstring = std::string("_leftImg8bit.png");
        m_extractBoundingboxes = true;
        m_hasSequence = false;
        m_fov = 120.0;
        break;
    case Mode::Test:
        m_groundTruthPath = basePath / bfs::path("gtFine") / bfs::path("test");
        m_leftImgPath = basePath / bfs::path("leftImg8bit") / bfs::path("test");
        m_prevLeftImgPath = basePath / bfs::path("leftImg8bit_sequence") / bfs::path("test");
        m_groundTruthSubstring = std::string("_gtFine_polygons.json");
        m_leftImgSubstring = std::string("_leftImg8bit.png");
        m_extractBoundingboxes = false;
        m_hasSequence = true;
        m_fov = 50.0;
        break;
    case Mode::Val:
        m_groundTruthPath = basePath / bfs::path("gtFine") / bfs::path("val");
        m_leftImgPath = basePath / bfs::path("leftImg8bit") / bfs::path("val");
        m_prevLeftImgPath = basePath / bfs::path("leftImg8bit_sequence") / bfs::path("val");
        m_groundTruthSubstring = std::string("_gtFine_polygons.json");
        m_leftImgSubstring = std::string("_leftImg8bit.png");
        m_extractBoundingboxes = true;
        m_hasSequence = true;
        m_fov = 50.0;
        break;
    default:
        CHECK(false, "Unknown mode!");
    }

    for (auto &entry : bfs::recursive_directory_iterator(m_groundTruthPath)) {
        if (entry.path().extension() == ".json") {
            auto relativePath = bfs::relative(entry.path(), m_groundTruthPath);
            std::string key = relativePath.string();
            key = key.substr(0, key.length() - m_groundTruthSubstring.length());
            m_keys.push_back(key);
        }
    }
    std::sort(m_keys.begin(), m_keys.end());
}

std::string CityscapesDataset::keyToPrev(std::string key) const
{
    if (!m_hasSequence) {
        return key;
    }

    std::string remainder = key;
    auto pos = remainder.find("_");
    std::string city = remainder.substr(0, pos);
    remainder = remainder.substr(pos + 1);
    pos = remainder.find("_");
    std::string imgno = remainder.substr(0, pos);
    std::string seqno = remainder.substr(pos + 1);

    /* We go two images back to have a larger displacement / a lower simulated frame rate */
    int numericSeqno = std::stoi(seqno) - 2;

    std::ostringstream result;
    result << city << "_" << imgno << "_" << std::setw(6) << std::setfill('0') << numericSeqno;

    return result.str();
}

std::shared_ptr<DatasetEntry> CityscapesDataset::get(std::size_t i)
{
    CHECK(i < m_keys.size(), "Index out of range");
    auto key = m_keys[i];
    auto result = std::make_shared<DatasetEntry>();
    auto leftImgPath = m_leftImgPath / bfs::path(key + m_leftImgSubstring);
    cv::Mat leftImg = cv::imread(leftImgPath.string());
    CHECK(leftImg.data, "Failed to read image " + leftImgPath.string());
    result->input.left = toFloatMat(leftImg);
    auto prevLeftImgPath = m_prevLeftImgPath / bfs::path(keyToPrev(key) + m_leftImgSubstring);
    cv::Mat prevLeftImg = cv::imread(prevLeftImgPath.string());
    CHECK(prevLeftImg.data, "Failed to read image " + prevLeftImgPath.string());
    result->input.prevLeft = toFloatMat(prevLeftImg);
    auto jsonPath = m_groundTruthPath / bfs::path(key + m_groundTruthSubstring);
    std::ifstream jsonFs(jsonPath.string());
    std::string jsonStr = std::string(std::istreambuf_iterator<char>(jsonFs), std::istreambuf_iterator<char>());
    auto [pixelwiseLabels, bbDontCareAreas, bbList] = parseJson(jsonStr, result->input.left.size());
    result->gt.pixelwiseLabels = pixelwiseLabels;
    result->gt.bbDontCareAreas = bbDontCareAreas;
    if (m_extractBoundingboxes) {
        result->gt.bbList = bbList;
    }
    result->metadata.originalWidth = result->input.left.cols;
    result->metadata.originalHeight = result->input.left.rows;
    result->metadata.canFlip = true;
    result->metadata.horizontalFov = m_fov;
    result->metadata.key = key;
    return result;
}

std::tuple<std::string, bool> CityscapesDataset::removeGroup(std::string label) const
{
    bool isGroup = boost::ends_with(label, "group");
    if (isGroup) {
        label.erase(label.end() - std::string("group").length(), label.end());
    }
    return {label, isGroup};
}

std::tuple<cv::Mat, cv::Mat, BoundingBoxList> CityscapesDataset::parseJson(const std::string jsonStr,
                                                                           cv::Size imageSize)
{
    Json::Value root;
    Json::Reader reader;
    bool success = reader.parse(jsonStr, root);
    CHECK(success, "Failed to parse JSON string");

    cv::Mat labelImg(imageSize, CV_32SC1, cv::Scalar(m_semanticDontCareLabel));
    cv::Mat bbDontCareImg(imageSize, CV_32SC1, cv::Scalar(m_boundingBoxDontCareLabel));
    BoundingBoxList bbList;
    bbList.valid = true;
    bbList.width = imageSize.width;
    bbList.height = imageSize.height;

    int64_t objectId = getRandomId();
    for (auto &annotation : root["objects"]) {
        if (annotation.get("deleted", 0).asInt() == 1) {
            continue;
        }
        auto [cls, isGroup] = removeGroup(annotation["label"].asString());
        std::vector<cv::Point> points;
        int xMin = std::numeric_limits<int>::max();
        int yMin = std::numeric_limits<int>::max();
        int xMax = std::numeric_limits<int>::lowest();
        int yMax = std::numeric_limits<int>::lowest();
        for (auto &point : annotation["polygon"]) {
            int x = point[0].asInt();
            int y = point[1].asInt();
            xMin = std::min(xMin, x);
            yMin = std::min(yMin, y);
            xMax = std::max(xMax, x);
            yMax = std::max(yMax, y);
            points.push_back(cv::Point(x, y));
        }
        int numPoints = points.size();
        const cv::Point* ppoints = points.data();
        CHECK(numPoints >= 3, "The object must contain at least 3 points");

        if (m_labelDict.count(cls) > 0) {
            /* Draw label image */
            cv::fillPoly(labelImg, &ppoints, &numPoints, 1, cv::Scalar(m_labelDict.at(cls)));

            /* Draw don't care image for bounding boxes.
             * We assume here that all don't care areas for pixel-wise labels are also
             * don't care areas for bounding boxes. This is not completely correct but
             * seems to work reasonably well. */
            if (!isGroup) {
                cv::fillPoly(bbDontCareImg, &ppoints, &numPoints, 1, cv::Scalar(m_boundingBoxValidLabel));
            }
        }

        /* Generate bounding box list */
        if (m_instanceDict.count(cls) > 0) {
            BoundingBox boundingBox;
            boundingBox.id = objectId++;
            boundingBox.cls = m_instanceDict.at(cls);
            boundingBox.x1 = xMin;
            boundingBox.x2 = xMax;
            boundingBox.y1 = yMin;
            boundingBox.y2 = yMax;
            bbList.boxes.push_back(boundingBox);
        }
    }

    return {labelImg, bbDontCareImg, bbList};
}
