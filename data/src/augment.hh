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

#include "data.hh"

#include <random>
#include <mutex>

class PointwiseDistort {
public:
    PointwiseDistort();
    void apply(std::shared_ptr<DatasetEntry> data);

private:
    void distortImage(cv::Mat &img, double gamma, double brightness, double contrast,
        double saturation, double hue, double deltaA, double deltaB, double blur, double noiseSigma);

    std::mutex m_generatorMutex;
    std::mt19937 m_generator;
    std::uniform_real_distribution<double> m_gammaDistribution {0.8, 1.2};
    std::uniform_real_distribution<double> m_brightnessDistribution {-0.2, 0.1};
    std::uniform_real_distribution<double> m_contrastDistribution {0.8, 1.2};
    std::uniform_real_distribution<double> m_saturationDistribution {-0.1, 0.1};
    std::uniform_real_distribution<double> m_hueDistribution {-10.0, 10.0};
    std::uniform_real_distribution<double> m_deltaColorDistribution {-5.0, 5.0};
    std::uniform_real_distribution<double> m_blurDistribution {0.0, 1.3};
    std::uniform_real_distribution<double> m_noiseSigmaDistribution {0.0, 0.03};
};

class CropResize {
public:
    CropResize(cv::Size newSize, double desiredHorizontalFov, bool allowFlip, double deltaCrop = 0.1);
    void apply(std::shared_ptr<DatasetEntry> data);

private:
    void handleInputImage(cv::Mat &img, cv::Rect roi, bool flip) const;
    void handleGtImage(cv::Mat &img, cv::Rect roi, bool flip) const;
    void handleBoundingBoxList(BoundingBoxList &boxList, cv::Rect roi, bool flip) const;

    cv::Size m_newSize;
    double m_desiredHorizontalFov;
    bool m_allowFlip;
    double m_deltaCrop;

    std::mutex m_generatorMutex;
    std::mt19937 m_generator;
    std::uniform_int_distribution<int> m_flipDistribution {0, 1};
};
