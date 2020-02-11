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

class WebvisionDataset : public FileDataset {
public:
    WebvisionDataset(bfs::path basePath);
    virtual std::shared_ptr<DatasetEntry> get(std::size_t i) override;

private:
    class Entry {
    public:
        std::string key;
        int cls;

        bool operator< (const Entry& other) const
        {
            return (key < other.key);
        }
    };

    bfs::path m_path;
    std::vector<Entry> m_entries;
};

