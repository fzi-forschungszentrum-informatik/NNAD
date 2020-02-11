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

#include "utils.hh"

#include "webvision_dataset.hh"

WebvisionDataset::WebvisionDataset(bfs::path basePath) : m_path(basePath)
{
    auto gtPath = m_path / bfs::path("info") / bfs::path("train_filelist.txt");
    std::ifstream gtfile(gtPath.string());
    std::string key;
    int cls;
    while (gtfile >> key >> cls) {
        key = key.substr(0, key.length() - std::string(".jpg").length());
        Entry entry;
        entry.key = key;
        entry.cls = cls;
        m_entries.push_back(entry);
        /* We don't use the keys. But we push them here anyway so that ::num() is happy. */
        m_keys.push_back(key);
    }
    std::sort(m_entries.begin(), m_entries.end());
}

std::shared_ptr<DatasetEntry> WebvisionDataset::get(std::size_t i)
{
    CHECK(i < m_entries.size(), "Index out of range");
    auto key = m_entries[i].key;
    auto result = std::make_shared<DatasetEntry>();

    auto imgPath = m_path / bfs::path(key + std::string(".jpg"));
    cv::Mat img = cv::imread(imgPath.string());
    CHECK(img.data, "Failed to read image " + imgPath.string());
    result->input.left = toFloatMat(img);

    result->gt.cls = m_entries[i].cls;

    result->metadata.originalWidth = result->input.left.cols;
    result->metadata.originalHeight = result->input.left.rows;
    result->metadata.canFlip = true;
    result->metadata.horizontalFov = -1.0;
    result->metadata.key = key;
    return result;
}
