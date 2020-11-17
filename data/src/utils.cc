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

#include <cstdlib>
#include <iostream>

#include "utils.hh"

void CHECK(const bool ok, const std::string errorMessage)
{
    if (!ok) {
        std::cerr << errorMessage << std::endl;
        std::abort();
    }
}

bool stringEndsWith(const std::string &str, const std::string &end)
{
    if (str.size() < end.size()) {
        return false;
    }

    if (str.compare(str.size() - end.size(), end.size(), end) == 0) {
        return true;
    }
    return false;
}

