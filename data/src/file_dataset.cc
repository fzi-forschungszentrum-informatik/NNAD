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

#include "utils.hh"
#include "file_dataset.hh"

#include <chrono>
#include <random>

FileDataset::FileDataset()
{
    auto seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    m_generator = std::mt19937_64(seed);
    m_idDistribution = std::uniform_int_distribution<int64_t>(1, std::numeric_limits<int64_t>::max() -
        std::numeric_limits<int32_t>::max());
}

std::size_t FileDataset::num() const
{
    return m_keys.size();
}

cv::Mat FileDataset::toFloatMat(cv::Mat input) const
{
    cv::Mat output;
    input.convertTo(output, CV_32FC3);
    output /= 255.0;
    return output;
}

int64_t FileDataset::getRandomId()
{
    return m_idDistribution(m_generator);
}
