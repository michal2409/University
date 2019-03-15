#ifndef HELPER_H_
#define HELPER_H_

#include <cstddef>
#include <cassert>

using AttackPower = int;
using Speed = int;
using ShieldPoints = int;

class Starship
{
 public:
    virtual void takeDamage(AttackPower damage) = 0;
    virtual size_t getSize() const = 0;
    virtual ~Starship() = default;
};

class StarshipAttacker : public virtual Starship
{
 public:
    StarshipAttacker(AttackPower attack);
    virtual AttackPower getAttackPower() const;

 protected:
    AttackPower attack;
};

class StarshipFlyer : public virtual Starship
{
 public:
    StarshipFlyer(Speed speed);
    virtual Speed getSpeed() const;

 protected:
    Speed speed;
};

class StarshipDefender : public virtual Starship
{
 public:
    StarshipDefender(ShieldPoints shield);

    virtual ShieldPoints getShield() const;
    virtual void takeDamage(AttackPower damage) override;
    virtual size_t getSize() const override;

 protected:
    ShieldPoints shield;
};

#endif  // HELPER_H_
