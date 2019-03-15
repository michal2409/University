#ifndef BATTLE_H_
#define BATTLE_H_

#include <iostream>
#include <cassert>
#include <vector>
#include <memory>

#include "imperialfleet.h"
#include "rebelfleet.h"

using Time = int;

class Strategy
{
 public:
    virtual bool isAttackTime() = 0;
    virtual void tickTime(Time timeStep) = 0;
    virtual ~Strategy() = default;
};

class Strategy1 : public Strategy
{
 public:
    Strategy1(Time startTime, Time maxTime);
    virtual bool isAttackTime();
    virtual void tickTime(Time timeStep);
 private:
    Time currTime;
    Time maxTime;
};

class SpaceBattle
{
 public:
    using strategyPtr = std::unique_ptr<Strategy>;
    using rebelPtr = std::shared_ptr<RebelStarship>;
    using imperialPtr = std::shared_ptr<ImperialStarship>;
    using rebelVect = std::vector<rebelPtr>;
    using imperialVect = std::vector<imperialPtr>;

    class Builder;

    SpaceBattle(Time startTime, Time maxTime, rebelVect rebels, imperialVect imperials,
                                             size_t rebelsCount, size_t imperialCount);

    size_t countImperialFleet() const;
    size_t countRebelFleet() const;
    void tick(Time timeStep);
    void battle();

 private:
    rebelVect rebels;
    imperialVect imperials;
    size_t rebelsCount;
    size_t imperialCount;
    strategyPtr strategy;
};

class SpaceBattle::Builder
{
 public:
    Builder& startTime(Time t);
    Builder& maxTime(Time t);
    Builder& ship(rebelPtr rebel);
    Builder& ship(imperialPtr imperial);
    SpaceBattle build();

 private:
    Time currTime;
    Time _maxTime;
    rebelVect rebels;
    imperialVect imperials;
    size_t rebelsCount = 0;
    size_t imperialCount = 0;
};

#endif  // BATTLE_H_
