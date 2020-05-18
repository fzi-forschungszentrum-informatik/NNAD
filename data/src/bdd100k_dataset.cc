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

#include "bdd100k_dataset.hh"

Bdd100kDataset::Bdd100kDataset(bfs::path basePath, Mode mode)
{
    switch (mode) {
    case Mode::TrainTracking:
        m_groundTruthPath = basePath / bfs::path("labels-20") / bfs::path("box-track") / bfs::path("train");
        m_leftImgPath = basePath / bfs::path("images") / bfs::path("track") / bfs::path("train");
        m_hasSequence = true;
        break;
    default:
        CHECK(false, "Unknown mode!");
    }

    for (auto &entry : bfs::recursive_directory_iterator(m_leftImgPath)) {
        if (entry.path().extension() == ".jpg") {
            std::string key = entry.path().filename().string();
            key = key.substr(0, key.length() - std::string(".jpg").length());

            /* Do not push the first image in the sequence, because it has no previous image */
            auto [prefix, seqNo] = splitKey(key);
            if (seqNo > 1) {
                m_keys.push_back(key);
            }
        }
    }
    std::sort(m_keys.begin(), m_keys.end());
}

std::tuple<std::string, int> Bdd100kDataset::splitKey(std::string key) const
{
    std::string remainder = key;
    auto pos = remainder.find("-");
    std::string id1 = remainder.substr(0, pos);
    remainder = remainder.substr(pos + 1);
    pos = remainder.find("-");
    std::string id2 = remainder.substr(0, pos);
    std::string seqno = remainder.substr(pos + 1);

    int numericSeqno = std::stoi(seqno);
    std::string prefix = id1 + std::string("-") + id2;

    return {prefix, numericSeqno};
}

std::string Bdd100kDataset::keyToPrev(std::string key) const
{
    if (!m_hasSequence) {
        return key;
    }

    auto [prefix, numericSeqno] = splitKey(key);

    /* Go to previous image */
    --numericSeqno;

    std::ostringstream result;
    result << prefix << "-" << std::setw(7) << std::setfill('0') << numericSeqno;

    return result.str();
}

std::shared_ptr<DatasetEntry> Bdd100kDataset::get(std::size_t i)
{
    CHECK(i < m_keys.size(), "Index out of range");
    auto key = m_keys[i];
    auto result = std::make_shared<DatasetEntry>();
    auto [keyPrefix, seqNo] = splitKey(key);
    auto leftImgPath = m_leftImgPath / bfs::path(keyPrefix) / bfs::path(key + std::string(".jpg"));
    cv::Mat leftImg = cv::imread(leftImgPath.string());
    CHECK(leftImg.data, "Failed to read image " + leftImgPath.string());
    result->input.left = toFloatMat(leftImg);
    auto prevLeftImgPath = m_leftImgPath / bfs::path(keyPrefix) / bfs::path(keyToPrev(key) + std::string(".jpg"));
    cv::Mat prevLeftImg = cv::imread(prevLeftImgPath.string());
    CHECK(prevLeftImg.data, "Failed to read image " + prevLeftImgPath.string());
    result->input.prevLeft = toFloatMat(prevLeftImg);
    auto jsonPath = m_groundTruthPath / bfs::path(keyPrefix + std::string(".json"));
    std::ifstream jsonFs(jsonPath.string());
    std::string jsonStr = std::string(std::istreambuf_iterator<char>(jsonFs), std::istreambuf_iterator<char>());
    auto bbList = parseJson(jsonStr, keyPrefix, seqNo, result->input.left.size());
    result->gt.bbList = bbList;
    result->gt.pixelwiseLabels = cv::Mat(result->input.left.size(), CV_32SC1, cv::Scalar(m_semanticDontCareLabel));
    result->metadata.originalWidth = result->input.left.cols;
    result->metadata.originalHeight = result->input.left.rows;
    result->metadata.canFlip = true;
    result->metadata.horizontalFov = 50.0; // This is just an estimate...
    result->metadata.key = key;
    return result;
}

BoundingBoxList Bdd100kDataset::parseJson(const std::string jsonStr, std::string keyPrefix, int seqNo, cv::Size imageSize) const
{
    Json::Value root;
    Json::Reader reader;
    bool success = reader.parse(jsonStr, root);
    CHECK(success, "Failed to parse JSON string");

    BoundingBoxList bbList;
    bbList.valid = true;
    bbList.previousValid = m_hasSequence;
    bbList.width = imageSize.width;
    bbList.height = imageSize.height;

    std::ostringstream currentName, previousName;
    currentName << keyPrefix << "-" << std::setw(7) << std::setfill('0') << seqNo << ".jpg";
    previousName << keyPrefix << "-" << std::setw(7) << std::setfill('0') << seqNo - 1 << ".jpg";

    Json::Value currentRoot;
    Json::Value previousRoot;

    for (auto &entry : root) {
        if (entry["name"] == currentName.str()) {
            currentRoot = entry["labels"];
        }
        if (entry["name"] == previousName.str()) {
            previousRoot = entry["labels"];
        }
    }

    for (auto &annotation : currentRoot) {
        std::string id = annotation["id"].asString();
        if (id.empty()) {
            std::cout << "Skipping annotation " << annotation << std::endl;
            continue;
        }
        std::string cls = annotation["category"].asString();
        int32_t xMin = static_cast<int32_t>(annotation["box2d"]["x1"].asDouble());
        int32_t xMax = static_cast<int32_t>(annotation["box2d"]["x2"].asDouble());
        int32_t yMin = static_cast<int32_t>(annotation["box2d"]["y1"].asDouble());
        int32_t yMax = static_cast<int32_t>(annotation["box2d"]["y2"].asDouble());

        int32_t oldXMin = 0;
        int32_t oldXMax = 0;
        int32_t oldYMin = 0;
        int32_t oldYMax = 0;
        bool hasPrevious = false;

        for (auto &previousAnnotation : previousRoot) {
            std::string previousId = previousAnnotation["id"].asString();
            if (previousId.empty()) {
                continue;
            }
            if (previousId == id) {
                hasPrevious = true;
                oldXMin = static_cast<int32_t>(previousAnnotation["box2d"]["x1"].asDouble());
                oldXMax = static_cast<int32_t>(previousAnnotation["box2d"]["x2"].asDouble());
                oldYMin = static_cast<int32_t>(previousAnnotation["box2d"]["y1"].asDouble());
                oldYMax = static_cast<int32_t>(previousAnnotation["box2d"]["y2"].asDouble());
            }
        }

        /* Generate bounding box list */
        if (m_instanceDict.count(cls) > 0) {
            BoundingBox boundingBox;
            boundingBox.cls = m_instanceDict.at(cls);
            boundingBox.id = std::stoi(id);
            boundingBox.x1 = xMin;
            boundingBox.x2 = xMax;
            boundingBox.y1 = yMin;
            boundingBox.y2 = yMax;
            if (hasPrevious) {
                BoundingBox previousBoundingBox;
                previousBoundingBox.cls = m_instanceDict.at(cls);
                previousBoundingBox.id = std::stoi(id);
                previousBoundingBox.x1 = oldXMin;
                previousBoundingBox.x2 = oldXMax;
                previousBoundingBox.y1 = oldYMin;
                previousBoundingBox.y2 = oldYMax;
                bbList.previousBoxes.push_back(previousBoundingBox);
                boundingBox.setDeltaFromPrevious(previousBoundingBox);
            }

            bbList.boxes.push_back(boundingBox);
        }
    }

    return bbList;
}

