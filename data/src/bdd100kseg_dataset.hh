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

class Bdd100kSegDataset : public FileDataset {
public:
    enum class Mode {
        Train = 0,
        Test,
        Val
    };

    Bdd100kSegDataset(bfs::path basePath, Mode mode);
    virtual std::shared_ptr<DatasetEntry> get(std::size_t i) override;

    bfs::path m_groundTruthPath;
    bfs::path m_leftImgPath;
    std::string m_groundTruthSubstring;
    std::string m_leftImgSubstring;
    double m_fov;

};
