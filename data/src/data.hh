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

#include <string>
#include <map>
#include <memory>
#include <variant>
#include <opencv2/core/mat.hpp>

struct BoundingBox {
    BoundingBox() : deltaValid(false)
    {
    }

    void setDeltaFromPrevious(BoundingBox &prevBox)
    {
        double width = x2 - x1;
        double prevWidth = prevBox.x2 - prevBox.x1;
        double height = y2 - y1;
        double prevHeight = prevBox.y2 - prevBox.y1;
        double xc = x1 + 0.5 * width;
        double prevXc = prevBox.x1 + 0.5 * prevWidth;
        double yc = y1 + 0.5 * height;
        double prevYc = prevBox.y1 + 0.5 * prevHeight;

        dxc = prevXc - xc;
        dyc = prevYc - yc;
        dw = std::log(prevWidth / width);
        dh = std::log(prevHeight / height);
        deltaValid = true;
    }

    int64_t id;
    int32_t cls;
    int32_t x1;
    int32_t y1;
    int32_t x2;
    int32_t y2;

    double dxc; /* xc_prev - xc_curr */
    double dyc; /* yc_prev - yc_curr */
    double dw; /* log(w_prev / w_curr) */
    double dh; /* log(h_prev / h_curr) */
    bool deltaValid;
};

struct BoundingBoxDetection {
    BoundingBox box;
    float score;
    std::vector<float> embedding;
};

struct AnchorBox {
    /* Box center */
    int32_t xc;
    int32_t yc;

    /* Top-left corner */
    int32_t x1;
    int32_t y1;

    /* Bottom-right corner */
    int32_t x2;
    int32_t y2;
};

enum Objectness {
    DONT_CARE = -1,
    NO_OBJECT = 0,
    OBJECT = 1
};

struct TargetBox {
    TargetBox() : id(0), dxc(0.0), dyc(0.0), dw(0.0), dh(0.0), deltaValid(0), deltaPrevXc(0.0), deltaPrevYc(0.0),
    deltaPrevW(0.0), deltaPrevH(0.0), cls(0), objectness(Objectness::NO_OBJECT)
    {
    }

    /* Box id */
    int64_t id;

    /* See Appendix C of "Rich feature hierarchies for accurate object detection and semantic segmentation" */
    double dxc;
    double dyc;
    double dw;
    double dh;

    /* Delta movement w.r.t. the box in the previous frame. */
    int32_t deltaValid;
    double deltaPrevXc;
    double deltaPrevYc;
    double deltaPrevW;
    double deltaPrevH;

    /* Class */
    int32_t cls;

    /* Objectness */
    int32_t objectness;
};

struct TargetBoxDetection {
    TargetBoxDetection() : dxc(0.0), dyc(0.0), dw(0.0), dh(0.0), cls(0), objectnessScore(0.0)
    {
    }

    /* See Appendix C of "Rich feature hierarchies for accurate object detection and semantic segmentation" */
    double dxc;
    double dyc;
    double dw;
    double dh;

    /* Delta movement w.r.t. the box in the previous frame. */
    double deltaPrevXc;
    double deltaPrevYc;
    double deltaPrevW;
    double deltaPrevH;

    /* Class */
    int32_t cls;

    std::vector<float> embedding;

    /* Objectness */
    double objectnessScore;
};

struct BoundingBoxList {
    BoundingBoxList() : width(0), height(0), valid(false), previousValid(false)
    {
    }

    int32_t width;
    int32_t height;
    bool valid;
    bool previousValid;
    std::vector<BoundingBox> boxes;
    std::vector<BoundingBox> previousBoxes;
    std::vector<std::vector<TargetBox>> targets;
    std::vector<std::vector<TargetBox>> previousTargets;
};

struct DatasetEntry {
    struct {
        cv::Mat left;
        cv::Mat prevLeft;
    } input;

    struct {
        cv::Mat flow;
        cv::Mat flowMask;
        cv::Mat pixelwiseLabels;
        cv::Mat bbDontCareAreas;
        BoundingBoxList bbList;
        int cls; // For pretraining
    } gt;

    struct {
        std::string key;
        int32_t originalWidth;
        int32_t originalHeight;
        bool canFlip;
        double horizontalFov;
    } metadata;
};
