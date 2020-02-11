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

#include "dataset.hh"

#include <boost/filesystem.hpp>
#include <boost/algorithm/string/predicate.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <json/json.h>

namespace bfs = boost::filesystem;

class FileDataset : public SequentialDataset {
public:
    FileDataset();
    std::size_t num() const override;

protected:
    cv::Mat toFloatMat(const cv::Mat input) const;
    int64_t getRandomId();

    std::vector<std::string> m_keys;

    const int32_t m_semanticDontCareLabel {-1};
    const int32_t m_boundingBoxDontCareLabel {1};
    const int32_t m_boundingBoxValidLabel {0};

private:
    std::mt19937_64 m_generator;
    std::uniform_int_distribution<int64_t> m_idDistribution;
};

