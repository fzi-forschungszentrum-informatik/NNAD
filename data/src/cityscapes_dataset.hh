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

#pragma once

#include "file_dataset.hh"

#include <boost/filesystem.hpp>
#include <boost/algorithm/string/predicate.hpp>
#include <json/json.h>

namespace bfs = boost::filesystem;

class CityscapesDataset : public FileDataset {
public:
    enum class Mode {
        Train = 0,
        TrainExtra,
        Test,
        Val
    };

    CityscapesDataset(bfs::path basePath, Mode mode);
    virtual std::shared_ptr<DatasetEntry> get(std::size_t i) override;

private:
    std::string keyToPrev(std::string key) const;
    std::tuple<std::string, bool> removeGroup(std::string label) const;
    std::tuple<cv::Mat, cv::Mat, BoundingBoxList> parseJson(const std::string jsonStr, cv::Size imageSize);

    bfs::path m_groundTruthPath;
    bfs::path m_leftImgPath;
    bfs::path m_prevLeftImgPath;
    std::string m_groundTruthSubstring;
    std::string m_leftImgSubstring;
    bool m_extractBoundingboxes;
    bool m_hasSequence;
    double m_fov;

    const std::map<std::string, int32_t> m_labelDict {
        {"road", 0},
        {"sidewalk", 1},
        {"building", 2},
        {"wall", 3},
        {"fence", 4},
        {"pole", 5},
        {"traffic light", 6},
        {"traffic sign", 7},
        {"vegetation", 8},
        {"terrain", 9},
        {"sky", 10},
        {"person", 11},
        {"rider", 12},
        {"car", 13},
        {"truck", 14},
        {"bus", 15},
        {"train", 16},
        {"motorcycle", 17},
        {"bicycle", 18},
    };

    const std::map<std::string, int32_t> m_instanceDict {
        {"person", 0},
        {"rider", 1},
        {"car", 2},
        {"truck", 3},
        {"bus", 4},
        {"train", 5},
        {"motorcycle", 6},
        {"bicycle", 7},
        {"traffic light", 8},
        {"traffic sign", 9},
    };
};
