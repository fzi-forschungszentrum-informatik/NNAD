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

#include <memory>
#include <vector>

#include "utils.hh"
#include "data.hh"

class BBUtils {
public:
    BBUtils(int width, int height, int scale);

    void targetsFromBBList(std::shared_ptr<DatasetEntry> ds) const;
    std::vector<BoundingBoxDetection> bbListFromTargets(VectorView<float>(objectnessScores),
                                                        VectorView<int64_t> objectClass, VectorView<float> regression,
                                                        VectorView<float> deltaRegression, VectorView<float> embedding,
                                                        int embeddingLength, double threshold) const;

    static void performNMS(std::vector<BoundingBoxDetection> &detectionList, double lowOverlapThreshold = 0.1,
                           double highOverlapThreshold = 0.9);

    std::size_t numAnchors() const;

private:
    std::vector<TargetBox> targetsFromSingleBBList(std::vector<BoundingBox> &boxes) const;
    std::vector<double> anchorIou(const BoundingBox &box) const;
    void setTargetsAsIgnore(std::shared_ptr<DatasetEntry> ds) const;
    void boxToTarget(const BoundingBox &box, const AnchorBox &anchor, TargetBox &target) const;
    void detectionToBox(const TargetBoxDetection &targetDetection, const AnchorBox &anchor,
                        BoundingBoxDetection &boxDetection) const;

    std::vector<double> m_ratios = {0.25, 0.5, 1.0, 2.0, 4.0};
    std::vector<int> m_areas = {12, 16, 24, 32};
    std::vector<AnchorBox> m_anchorBoxes;

    int m_width;
    int m_height;
    int m_scale;
};
