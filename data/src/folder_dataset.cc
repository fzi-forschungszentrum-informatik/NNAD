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

#include "folder_dataset.hh"

FolderDataset::FolderDataset(bfs::path path)
{
    m_leftImgPath = path;

    for (auto &entry : bfs::recursive_directory_iterator(m_leftImgPath)) {
        if (entry.path().extension() == ".jpg" || entry.path().extension() == ".png") {
            auto relativePath = bfs::relative(entry.path(), m_leftImgPath);
            std::string key = relativePath.string();
            key = key.substr(0, key.length() - entry.path().extension().string().length());
            m_keys.push_back(key);
            m_fileEnding[key] = entry.path().extension().string();
        }
    }
    std::sort(m_keys.begin(), m_keys.end());
}

std::shared_ptr<DatasetEntry> FolderDataset::get(std::size_t i)
{
    CHECK(i < m_keys.size(), "Index out of range");
    auto key = m_keys[i];
    auto result = std::make_shared<DatasetEntry>();
    auto leftImgPath = m_leftImgPath / bfs::path(key + m_fileEnding[key]);
    cv::Mat leftImg = cv::imread(leftImgPath.string());
    CHECK(leftImg.data, "Failed to read image " + leftImgPath.string());
    result->input.left = toFloatMat(leftImg);
    result->metadata.originalWidth = result->input.left.cols;
    result->metadata.originalHeight = result->input.left.rows;
    result->metadata.canFlip = false;
    result->metadata.horizontalFov = -1.0;
    result->metadata.key = key;
    return result;
}
