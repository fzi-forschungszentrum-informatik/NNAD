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

class Bdd100kDataset : public FileDataset {
public:
    enum class Mode {
        TrainTracking = 0,
    };

    Bdd100kDataset(bfs::path basePath, Mode mode);
    virtual std::shared_ptr<DatasetEntry> get(std::size_t i) override;

private:
    std::tuple<std::string, int> splitKey(std::string key) const;
    std::string keyToPrev(std::string key) const;
    BoundingBoxList parseJson(const std::string jsonStr, std::string keyPrefix, int seqNo, cv::Size imageSize) const;

    bfs::path m_groundTruthPath;
    bfs::path m_leftImgPath;
    bool m_hasSequence;

    const std::map<std::string, int32_t> m_instanceDict { //TODO
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

