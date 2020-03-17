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

#include <random>

namespace bfs = boost::filesystem;

class SintelDataset : public FileDataset {
public:
    SintelDataset(bfs::path basePath);
    virtual std::shared_ptr<DatasetEntry> get(std::size_t i) override;

private:
    cv::Mat readFlowImg(std::string filename) const;
    std::vector<cv::Mat> flowPyramid(cv::Mat flow) const;
    std::string nextKey(std::string &key) const;

    bfs::path m_flowPath;
    bfs::path m_imagePath;

    std::mt19937 m_generator;
    std::uniform_int_distribution<int> m_yDistribution {0, 436 - 384};
};

