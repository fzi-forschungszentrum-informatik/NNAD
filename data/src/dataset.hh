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

#include <functional>
#include <random>
#include <queue>
#include <thread>
#include <condition_variable>
#include <variant>
#include <mutex>

class Dataset {
public:
    virtual std::shared_ptr<DatasetEntry> getNext() = 0;
};

class SequentialDataset : public Dataset {
public:
    virtual std::shared_ptr<DatasetEntry> get(std::size_t i) = 0;
    virtual std::size_t num() const = 0;
    virtual std::shared_ptr<DatasetEntry> getNext() override;

private:
    std::mutex m_idxMutex;
    std::size_t m_idx {0};
};


class ParallelDataset : public Dataset {
public:
    ParallelDataset(std::shared_ptr<Dataset> inputDataset, int numWorkers = 16, std::size_t queueSize = 64);
    ~ParallelDataset();

    virtual std::shared_ptr<DatasetEntry> getNext() override;

private:
    void worker();

    std::shared_ptr<Dataset> m_inputDataset;
    std::vector<std::thread> m_threads;
    std::size_t m_queueSize;
    bool m_run;
    int m_numWorkers;
    std::mutex m_numWorkersMutex;
    std::condition_variable m_pushCondVar;
    std::condition_variable m_popCondVar;
    std::mutex m_mutex;
    std::mutex m_accessMutex;
    std::queue<std::shared_ptr<DatasetEntry>> m_queue;
};

class RandomDataset : public Dataset {
public:
    RandomDataset(std::shared_ptr<SequentialDataset> inputDataset);

    virtual std::shared_ptr<DatasetEntry> getNext() override;

private:
    std::shared_ptr<SequentialDataset> m_inputDataset;
    std::mutex m_generatorMutex;
    std::mt19937 m_generator;
    std::uniform_int_distribution<std::size_t> m_indexDistribution;
};

class InterleaveDataset : public Dataset {
public:
    InterleaveDataset(std::vector<std::shared_ptr<Dataset>> inputDatasets);

    virtual std::shared_ptr<DatasetEntry> getNext() override;

private:
    std::mutex m_datasetMutex;
    std::vector<std::shared_ptr<Dataset>> m_inputDatasets;
    std::mutex m_generatorMutex;
    std::mt19937 m_generator;
    std::uniform_int_distribution<std::size_t> m_indexDistribution;
};

class MapDataset : public Dataset {
public:
    MapDataset(std::shared_ptr<Dataset> inputDataset, std::function<void(std::shared_ptr<DatasetEntry>)> mapFn);

    virtual std::shared_ptr<DatasetEntry> getNext() override;

private:
    std::shared_ptr<Dataset> m_inputDataset;
    std::function<void(std::shared_ptr<DatasetEntry>)> m_mapFn;
};
