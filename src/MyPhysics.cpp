#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/math.hpp>
#include <godot_cpp/variant/utility_functions.hpp>
#include <godot_cpp/classes/physics_server3d.hpp>
#include <godot_cpp/classes/physics_test_motion_parameters3d.hpp>
#include <godot_cpp/classes/physics_test_motion_result3d.hpp>

#include <cstdint>
#include <cstring>

using namespace godot;

// --- 定义跨语言的 POD (Plain Old Data) 结构体 ---
// 必须保证 C++ 和 C# 的内存布局完全一致

#pragma pack(push, 1)

struct MoveParams {
    // Transform (3x4 matrix usually, or 4 vectors)
    // 这里我们拆解为 Basis(3x3) + Origin(1x3)
    float t_basis_xx, t_basis_xy, t_basis_xz;
    float t_basis_yx, t_basis_yy, t_basis_yz;
    float t_basis_zx, t_basis_zy, t_basis_zz;
    float t_origin_x, t_origin_y, t_origin_z;

    float vel_x, vel_y, vel_z;
    float up_x, up_y, up_z;

    float delta;
    float floor_max_angle;
    float floor_snap_length;
    float safe_margin;
    float wall_min_slide_angle;

    int32_t max_slides;

    uint8_t motion_mode; // 0=grounded, 1=floating
    uint8_t floor_stop_on_slope;
    uint8_t floor_constant_speed;
    uint8_t floor_block_on_wall;
    uint8_t slide_on_ceiling;
    uint8_t was_on_floor;
    uint8_t locked_axis; // bitmask: x=1, y=2, z=4
    uint8_t reserved0;

    float prev_floor_normal_x, prev_floor_normal_y, prev_floor_normal_z;

    uint64_t body_rid_id; // RID 的内部 ID
};

struct MoveResult {
    float new_pos_x, new_pos_y, new_pos_z;
    float new_vel_x, new_vel_y, new_vel_z;
    float real_vel_x, real_vel_y, real_vel_z;

    uint8_t on_floor;
    uint8_t on_wall;
    uint8_t on_ceiling;
    uint8_t reserved0;

    float floor_normal_x, floor_normal_y, floor_normal_z;
    float wall_normal_x, wall_normal_y, wall_normal_z;
    float ceiling_normal_x, ceiling_normal_y, ceiling_normal_z;

    float last_motion_x, last_motion_y, last_motion_z;
};

#pragma pack(pop)

#ifdef _WIN32
#define EXPORT_API __declspec(dllexport)
#else
#define EXPORT_API __attribute__((visibility("default")))
#endif

namespace {

constexpr real_t kFloorAngleThreshold = 0.01;
constexpr real_t kCmpEpsilon = 1e-5;
constexpr real_t kCancelSlidePrecision = 0.001;

enum MotionMode : uint8_t {
    MOTION_MODE_GROUNDED = 0,
    MOTION_MODE_FLOATING = 1,
};

struct CollisionState {
    bool floor = false;
    bool wall = false;
    bool ceiling = false;
};

struct MotionTestResult {
    Vector3 travel;
    Vector3 remainder;
    real_t collision_safe_fraction = 0.0;
    real_t collision_unsafe_fraction = 0.0;
    int collision_count = 0;
    Ref<PhysicsTestMotionResult3D> raw;
};

struct MoveContext {
    Transform3D transform;
    Vector3 velocity;
    Vector3 up_direction;
    real_t delta = 0.0;
    real_t floor_max_angle = 0.0;
    real_t floor_snap_length = 0.0;
    real_t margin = 0.0;
    real_t wall_min_slide_angle = 0.0;
    int max_slides = 1;
    bool floor_stop_on_slope = false;
    bool floor_constant_speed = false;
    bool floor_block_on_wall = false;
    bool slide_on_ceiling = false;
    MotionMode motion_mode = MOTION_MODE_GROUNDED;
    bool was_on_floor = false;
    uint8_t locked_axis = 0;
    Vector3 prev_floor_normal;
    CollisionState collision_state;
    Vector3 floor_normal;
    Vector3 wall_normal;
    Vector3 ceiling_normal;
    Vector3 platform_ceiling_velocity;
    Vector3 last_motion;
};

// 必须在物理线程调用；thread_local 缓存用于避免每次分配。
thread_local Ref<PhysicsTestMotionParameters3D> tl_params;
thread_local Ref<PhysicsTestMotionResult3D> tl_result;

static inline void ensure_thread_local_refs(Ref<PhysicsTestMotionParameters3D> &params, Ref<PhysicsTestMotionResult3D> &res) {
    if (tl_params.is_null()) {
        tl_params.instantiate();
    }
    if (tl_result.is_null()) {
        tl_result.instantiate();
    }
    params = tl_params;
    res = tl_result;
}

static inline real_t min_real(real_t a, real_t b) {
    return a < b ? a : b;
}

static inline real_t max_real(real_t a, real_t b) {
    return a > b ? a : b;
}

static inline real_t clamp_unit(real_t v) {
    if (v < -1.0) {
        return -1.0;
    }
    if (v > 1.0) {
        return 1.0;
    }
    return v;
}

static inline real_t safe_acos(real_t v) {
    return Math::acos(clamp_unit(v));
}

static inline void apply_axis_locks(Vector3 &vec, uint8_t mask) {
    if (mask & 1) {
        vec.x = 0.0;
    }
    if (mask & 2) {
        vec.y = 0.0;
    }
    if (mask & 4) {
        vec.z = 0.0;
    }
}

static inline CollisionState collision_state_mask(bool floor, bool wall, bool ceiling) {
    CollisionState s;
    s.floor = floor;
    s.wall = wall;
    s.ceiling = ceiling;
    return s;
}

static inline RID rid_from_uint64(uint64_t id) {
    RID rid;
    std::memcpy(rid._native_ptr(), &id, sizeof(id));
    return rid;
}

static bool move_and_collide(PhysicsServer3D *ps, const RID &body_rid, MoveContext &ctx, const Vector3 &motion, bool test_only, bool cancel_sliding, int max_collisions, bool collide_separation_ray, MotionTestResult &out, Ref<PhysicsTestMotionParameters3D> &params, Ref<PhysicsTestMotionResult3D> &res) {
    params->set_from(ctx.transform);
    params->set_motion(motion);
    params->set_margin(ctx.margin);
    params->set_max_collisions(max_collisions);
    params->set_recovery_as_collision_enabled(true);
    params->set_collide_separation_ray_enabled(collide_separation_ray);

    bool colliding = ps->body_test_motion(body_rid, params, res);

    out.raw = res;
    out.collision_count = res->get_collision_count();
    out.collision_safe_fraction = res->get_collision_safe_fraction();
    out.collision_unsafe_fraction = res->get_collision_unsafe_fraction();
    out.travel = res->get_travel();
    out.remainder = res->get_remainder();

    if (cancel_sliding) {
        real_t motion_length = motion.length();
        real_t precision = kCancelSlidePrecision;

        if (colliding && out.collision_count > 0) {
            precision += motion_length * (out.collision_unsafe_fraction - out.collision_safe_fraction);
            if (res->get_collision_depth(0) > ctx.margin + precision) {
                cancel_sliding = false;
            }
        }

        if (cancel_sliding) {
            Vector3 motion_normal;
            if (motion_length > kCmpEpsilon) {
                motion_normal = motion / motion_length;
            }

            real_t projected_length = out.travel.dot(motion_normal);
            Vector3 recovery = out.travel - motion_normal * projected_length;
            real_t recovery_length = recovery.length();
            if (recovery_length < ctx.margin + precision) {
                out.travel = motion_normal * projected_length;
                out.remainder = motion - out.travel;
            }
        }
    }

    apply_axis_locks(out.travel, ctx.locked_axis);

    if (!test_only) {
        ctx.transform.origin += out.travel;
    }

    return colliding;
}

static void set_collision_direction(const MotionTestResult &result, MoveContext &ctx, CollisionState &out_state, CollisionState apply_state) {
    out_state = CollisionState();

    real_t wall_depth = -1.0;
    real_t floor_depth = -1.0;

    bool was_on_wall = ctx.collision_state.wall;
    Vector3 prev_wall_normal = ctx.wall_normal;
    int wall_collision_count = 0;
    Vector3 combined_wall_normal;
    Vector3 tmp_wall_col;

    for (int i = result.collision_count - 1; i >= 0; --i) {
        Vector3 normal = result.raw->get_collision_normal(i);
        real_t depth = result.raw->get_collision_depth(i);

        if (ctx.motion_mode == MOTION_MODE_GROUNDED && !ctx.up_direction.is_zero_approx()) {
            real_t floor_angle = safe_acos(normal.dot(ctx.up_direction));
            if (floor_angle <= ctx.floor_max_angle + kFloorAngleThreshold) {
                out_state.floor = true;
                if (apply_state.floor && depth > floor_depth) {
                    ctx.collision_state.floor = true;
                    ctx.floor_normal = normal;
                    floor_depth = depth;
                }
                continue;
            }

            real_t ceiling_angle = safe_acos(normal.dot(-ctx.up_direction));
            if (ceiling_angle <= ctx.floor_max_angle + kFloorAngleThreshold) {
                out_state.ceiling = true;
                if (apply_state.ceiling) {
                    ctx.collision_state.ceiling = true;
                    ctx.ceiling_normal = normal;
                    ctx.platform_ceiling_velocity = result.raw->get_collider_velocity(i);
                }
                continue;
            }
        }

        out_state.wall = true;

        if (apply_state.wall && depth > wall_depth) {
            ctx.collision_state.wall = true;
            wall_depth = depth;
            ctx.wall_normal = normal;
        }

        if (!normal.is_equal_approx(tmp_wall_col)) {
            tmp_wall_col = normal;
            combined_wall_normal += normal;
            wall_collision_count++;
        }
    }

    if (out_state.wall && wall_collision_count > 1 && !out_state.floor && ctx.motion_mode == MOTION_MODE_GROUNDED) {
        if (combined_wall_normal != Vector3()) {
            combined_wall_normal.normalize();
            real_t floor_angle = safe_acos(combined_wall_normal.dot(ctx.up_direction));
            if (floor_angle <= ctx.floor_max_angle + kFloorAngleThreshold) {
                out_state.floor = true;
                out_state.wall = false;
                if (apply_state.floor) {
                    ctx.collision_state.floor = true;
                    ctx.floor_normal = combined_wall_normal;
                }
                if (apply_state.wall) {
                    ctx.collision_state.wall = was_on_wall;
                    ctx.wall_normal = prev_wall_normal;
                }
                return;
            }
        }
    }
}

static void apply_floor_snap(PhysicsServer3D *ps, const RID &body_rid, MoveContext &ctx, Ref<PhysicsTestMotionParameters3D> &params, Ref<PhysicsTestMotionResult3D> &res) {
    if (ctx.collision_state.floor || ctx.up_direction.is_zero_approx()) {
        return;
    }

    real_t length = max_real(ctx.floor_snap_length, ctx.margin);

    MotionTestResult result;
    bool collided = move_and_collide(ps, body_rid, ctx, -ctx.up_direction * length, true, false, 4, true, result, params, res);
    if (!collided) {
        return;
    }

    CollisionState result_state;
    set_collision_direction(result, ctx, result_state, collision_state_mask(true, false, false));

    if (result_state.floor) {
        if (result.travel.length() > ctx.margin) {
            result.travel = ctx.up_direction * ctx.up_direction.dot(result.travel);
        } else {
            result.travel = Vector3();
        }

        ctx.transform.origin += result.travel;
    }
}

static bool on_floor_if_snapped(PhysicsServer3D *ps, const RID &body_rid, MoveContext &ctx, Ref<PhysicsTestMotionParameters3D> &params, Ref<PhysicsTestMotionResult3D> &res, bool vel_dir_facing_up) {
    if (ctx.up_direction.is_zero_approx() || ctx.collision_state.floor || !ctx.was_on_floor || vel_dir_facing_up) {
        return false;
    }

    real_t length = max_real(ctx.floor_snap_length, ctx.margin);

    MotionTestResult result;
    if (move_and_collide(ps, body_rid, ctx, -ctx.up_direction * length, true, false, 4, true, result, params, res)) {
        CollisionState result_state;
        set_collision_direction(result, ctx, result_state, collision_state_mask(false, false, false));
        return result_state.floor;
    }

    return false;
}

static void snap_on_floor(PhysicsServer3D *ps, const RID &body_rid, MoveContext &ctx, Ref<PhysicsTestMotionParameters3D> &params, Ref<PhysicsTestMotionResult3D> &res, bool vel_dir_facing_up) {
    if (ctx.collision_state.floor || !ctx.was_on_floor || vel_dir_facing_up) {
        return;
    }

    apply_floor_snap(ps, body_rid, ctx, params, res);
}

static void move_and_slide_grounded(PhysicsServer3D *ps, const RID &body_rid, MoveContext &ctx, Ref<PhysicsTestMotionParameters3D> &params, Ref<PhysicsTestMotionResult3D> &res) {
    Vector3 motion = ctx.velocity * ctx.delta;
    Vector3 motion_slide_up = motion.slide(ctx.up_direction);
    Vector3 prev_floor_normal = ctx.prev_floor_normal;

    ctx.floor_normal = Vector3();
    ctx.wall_normal = Vector3();
    ctx.ceiling_normal = Vector3();
    ctx.platform_ceiling_velocity = Vector3();

    bool sliding_enabled = !ctx.floor_stop_on_slope;
    bool can_apply_constant_speed = sliding_enabled;
    bool apply_ceiling_velocity = false;
    bool first_slide = true;
    bool vel_dir_facing_up = ctx.velocity.dot(ctx.up_direction) > 0.0;
    Vector3 total_travel;

    for (int iteration = 0; iteration < ctx.max_slides; ++iteration) {
        MotionTestResult result;
        bool collided = move_and_collide(ps, body_rid, ctx, motion, false, !sliding_enabled, 6, false, result, params, res);

        ctx.last_motion = result.travel;

        if (collided) {
            CollisionState previous_state = ctx.collision_state;

            CollisionState result_state;
            set_collision_direction(result, ctx, result_state, collision_state_mask(true, true, true));

            if (ctx.collision_state.ceiling && ctx.platform_ceiling_velocity != Vector3() && ctx.platform_ceiling_velocity.dot(ctx.up_direction) < 0.0) {
                if (!ctx.slide_on_ceiling || motion.dot(ctx.up_direction) < 0.0 || (ctx.ceiling_normal + ctx.up_direction).length() < 0.01) {
                    apply_ceiling_velocity = true;
                    Vector3 ceiling_vertical_velocity = ctx.up_direction * ctx.up_direction.dot(ctx.platform_ceiling_velocity);
                    Vector3 motion_vertical_velocity = ctx.up_direction * ctx.up_direction.dot(ctx.velocity);
                    if (motion_vertical_velocity.dot(ctx.up_direction) > 0.0 || ceiling_vertical_velocity.length_squared() > motion_vertical_velocity.length_squared()) {
                        ctx.velocity = ceiling_vertical_velocity + ctx.velocity.slide(ctx.up_direction);
                    }
                }
            }

            if (ctx.collision_state.floor && ctx.floor_stop_on_slope && (ctx.velocity.normalized() + ctx.up_direction).length() < 0.01) {
                if (result.travel.length() <= ctx.margin + kCmpEpsilon) {
                    ctx.transform.origin -= result.travel;
                }
                ctx.velocity = Vector3();
                motion = Vector3();
                ctx.last_motion = Vector3();
                break;
            }

            if (result.remainder.is_zero_approx()) {
                motion = Vector3();
                break;
            }

            bool apply_default_sliding = true;

            if (result_state.wall && (motion_slide_up.dot(ctx.wall_normal) <= 0.0)) {
                if (ctx.floor_block_on_wall) {
                    Vector3 horizontal_motion = motion.slide(ctx.up_direction);
                    Vector3 horizontal_normal = ctx.wall_normal.slide(ctx.up_direction).normalized();
                    real_t motion_angle = Math::abs(safe_acos(-horizontal_normal.dot(horizontal_motion.normalized())));

                    if (motion_angle < 0.5 * Math_PI) {
                        apply_default_sliding = false;
                        if (ctx.was_on_floor && !vel_dir_facing_up) {
                            real_t travel_total = result.travel.length();
                            real_t cancel_dist_max = min_real(0.1, ctx.margin * 20.0);
                            if (travel_total <= ctx.margin + kCmpEpsilon) {
                                ctx.transform.origin -= result.travel;
                                result.travel = Vector3();
                            } else if (travel_total < cancel_dist_max) {
                                ctx.transform.origin -= result.travel.slide(ctx.up_direction);
                                motion = motion.slide(ctx.up_direction);
                                result.travel = Vector3();
                            } else {
                                result.travel = result.travel.slide(ctx.up_direction);
                                motion = result.remainder;
                            }

                            snap_on_floor(ps, body_rid, ctx, params, res, false);
                        } else {
                            motion = result.remainder;
                        }

                        Vector3 forward = ctx.wall_normal.slide(ctx.up_direction).normalized();
                        motion = motion.slide(forward);

                        if (vel_dir_facing_up) {
                            Vector3 slide_motion = ctx.velocity.slide(result.raw->get_collision_normal(0));
                            ctx.velocity = ctx.up_direction * ctx.up_direction.dot(ctx.velocity) + slide_motion.slide(ctx.up_direction);
                        } else {
                            ctx.velocity = ctx.velocity.slide(forward);
                        }

                        if (ctx.was_on_floor && !vel_dir_facing_up && (motion.dot(ctx.up_direction) > 0.0)) {
                            Vector3 floor_side = prev_floor_normal.cross(ctx.wall_normal);
                            if (floor_side != Vector3()) {
                                motion = floor_side * motion.dot(floor_side);
                            }
                        }

                        bool stop_all_motion = previous_state.wall && !vel_dir_facing_up;

                        if (!ctx.collision_state.floor && motion.dot(ctx.up_direction) < 0.0) {
                            Vector3 slide_motion = motion.slide(ctx.wall_normal);
                            if (slide_motion.dot(ctx.up_direction) < 0.0) {
                                stop_all_motion = false;
                                motion = slide_motion;
                            }
                        }

                        if (stop_all_motion) {
                            motion = Vector3();
                            ctx.velocity = Vector3();
                        }
                    }
                }

                if (ctx.was_on_floor && (ctx.wall_min_slide_angle > 0.0) && result_state.wall) {
                    Vector3 horizontal_normal = ctx.wall_normal.slide(ctx.up_direction).normalized();
                    real_t motion_angle = Math::abs(safe_acos(-horizontal_normal.dot(motion_slide_up.normalized())));
                    if (motion_angle < ctx.wall_min_slide_angle) {
                        motion = ctx.up_direction * motion.dot(ctx.up_direction);
                        ctx.velocity = ctx.up_direction * ctx.up_direction.dot(ctx.velocity);

                        apply_default_sliding = false;
                    }
                }
            }

            if (apply_default_sliding) {
                if ((sliding_enabled || !ctx.collision_state.floor) && (!ctx.collision_state.ceiling || ctx.slide_on_ceiling || !vel_dir_facing_up) && !apply_ceiling_velocity) {
                    Vector3 collision_normal = result.raw->get_collision_normal(0);

                    Vector3 slide_motion = result.remainder.slide(collision_normal);
                    if (ctx.collision_state.floor && !ctx.collision_state.wall && !motion_slide_up.is_zero_approx()) {
                        real_t motion_length = slide_motion.length();
                        slide_motion = ctx.up_direction.cross(result.remainder).cross(ctx.floor_normal);

                        slide_motion.normalize();
                        slide_motion *= motion_length;
                    }

                    if (slide_motion.dot(ctx.velocity) > 0.0) {
                        motion = slide_motion;
                    } else {
                        motion = Vector3();
                    }

                    if (ctx.slide_on_ceiling && result_state.ceiling) {
                        if (vel_dir_facing_up) {
                            ctx.velocity = ctx.velocity.slide(collision_normal);
                        } else {
                            ctx.velocity = ctx.up_direction * ctx.up_direction.dot(ctx.velocity);
                        }
                    }
                } else {
                    motion = result.remainder;
                    if (result_state.ceiling && !ctx.slide_on_ceiling && vel_dir_facing_up) {
                        ctx.velocity = ctx.velocity.slide(ctx.up_direction);
                        motion = motion.slide(ctx.up_direction);
                    }
                }
            }

            total_travel += result.travel;

            if (ctx.was_on_floor && ctx.floor_constant_speed && can_apply_constant_speed && ctx.collision_state.floor && !motion.is_zero_approx()) {
                Vector3 travel_slide_up = total_travel.slide(ctx.up_direction);
                motion = motion.normalized() * max_real(0.0, (motion_slide_up.length() - travel_slide_up.length()));
            }
        } else if (ctx.floor_constant_speed && first_slide && on_floor_if_snapped(ps, body_rid, ctx, params, res, vel_dir_facing_up)) {
            can_apply_constant_speed = false;
            sliding_enabled = true;
            ctx.transform.origin -= result.travel;

            Vector3 motion_slide_norm = ctx.up_direction.cross(motion).cross(prev_floor_normal);
            motion_slide_norm.normalize();

            motion = motion_slide_norm * (motion_slide_up.length());
            collided = true;
        }

        if (!collided || motion.is_zero_approx()) {
            break;
        }

        can_apply_constant_speed = !can_apply_constant_speed && !sliding_enabled;
        sliding_enabled = true;
        first_slide = false;
    }

    snap_on_floor(ps, body_rid, ctx, params, res, vel_dir_facing_up);

    if (ctx.collision_state.floor && !vel_dir_facing_up) {
        ctx.velocity = ctx.velocity.slide(ctx.up_direction);
    }
}

static void move_and_slide_floating(PhysicsServer3D *ps, const RID &body_rid, MoveContext &ctx, Ref<PhysicsTestMotionParameters3D> &params, Ref<PhysicsTestMotionResult3D> &res) {
    Vector3 motion = ctx.velocity * ctx.delta;

    ctx.floor_normal = Vector3();
    ctx.platform_ceiling_velocity = Vector3();

    bool first_slide = true;
    for (int iteration = 0; iteration < ctx.max_slides; ++iteration) {
        MotionTestResult result;
        bool collided = move_and_collide(ps, body_rid, ctx, motion, false, false, 1, false, result, params, res);

        ctx.last_motion = result.travel;

        if (collided) {
            CollisionState result_state;
            set_collision_direction(result, ctx, result_state, collision_state_mask(true, true, true));

            if (result.remainder.is_zero_approx()) {
                motion = Vector3();
                break;
            }

            if (ctx.wall_min_slide_angle != 0.0 && safe_acos(ctx.wall_normal.dot(-ctx.velocity.normalized())) < ctx.wall_min_slide_angle + kFloorAngleThreshold) {
                motion = Vector3();
                if (result.travel.length() < ctx.margin + kCmpEpsilon) {
                    ctx.transform.origin -= result.travel;
                }
            } else if (first_slide) {
                Vector3 motion_slide_norm = result.remainder.slide(ctx.wall_normal).normalized();
                motion = motion_slide_norm * (motion.length() - result.travel.length());
            } else {
                motion = result.remainder.slide(ctx.wall_normal);
            }

            if (motion.dot(ctx.velocity) <= 0.0) {
                motion = Vector3();
            }
        }

        if (!collided || motion.is_zero_approx()) {
            break;
        }

        first_slide = false;
    }
}

} // namespace

extern "C" {

EXPORT_API void c_move_and_slide(const MoveParams *input, MoveResult *output) {
    if (input == nullptr || output == nullptr) {
        return;
    }

    Basis basis(
        Vector3(input->t_basis_xx, input->t_basis_xy, input->t_basis_xz),
        Vector3(input->t_basis_yx, input->t_basis_yy, input->t_basis_yz),
        Vector3(input->t_basis_zx, input->t_basis_zy, input->t_basis_zz)
    );
    Vector3 origin(input->t_origin_x, input->t_origin_y, input->t_origin_z);
    Transform3D current_transform(basis, origin);

    Vector3 velocity(input->vel_x, input->vel_y, input->vel_z);
    Vector3 up_dir(input->up_x, input->up_y, input->up_z);

    MoveContext ctx;
    ctx.transform = current_transform;
    ctx.velocity = velocity;
    ctx.delta = static_cast<real_t>(input->delta);
    ctx.floor_max_angle = static_cast<real_t>(input->floor_max_angle);
    ctx.floor_snap_length = static_cast<real_t>(input->floor_snap_length);
    ctx.margin = static_cast<real_t>(input->safe_margin);
    ctx.wall_min_slide_angle = static_cast<real_t>(input->wall_min_slide_angle);
    ctx.max_slides = input->max_slides > 0 ? input->max_slides : 1;
    ctx.floor_stop_on_slope = input->floor_stop_on_slope != 0;
    ctx.floor_constant_speed = input->floor_constant_speed != 0;
    ctx.floor_block_on_wall = input->floor_block_on_wall != 0;
    ctx.slide_on_ceiling = input->slide_on_ceiling != 0;
    ctx.was_on_floor = input->was_on_floor != 0;
    ctx.locked_axis = input->locked_axis;
    ctx.prev_floor_normal = Vector3(input->prev_floor_normal_x, input->prev_floor_normal_y, input->prev_floor_normal_z);
    ctx.motion_mode = (input->motion_mode == MOTION_MODE_FLOATING) ? MOTION_MODE_FLOATING : MOTION_MODE_GROUNDED;

    if (!up_dir.is_zero_approx()) {
        ctx.up_direction = up_dir.normalized();
    } else {
        ctx.up_direction = Vector3();
        ctx.motion_mode = MOTION_MODE_FLOATING;
    }

    apply_axis_locks(ctx.velocity, ctx.locked_axis);

    PhysicsServer3D *ps = PhysicsServer3D::get_singleton();
    if (ps == nullptr) {
        return;
    }

    if (ctx.delta <= 0.0) {
        output->new_pos_x = ctx.transform.origin.x;
        output->new_pos_y = ctx.transform.origin.y;
        output->new_pos_z = ctx.transform.origin.z;

        output->new_vel_x = ctx.velocity.x;
        output->new_vel_y = ctx.velocity.y;
        output->new_vel_z = ctx.velocity.z;

        output->real_vel_x = 0.0;
        output->real_vel_y = 0.0;
        output->real_vel_z = 0.0;

        output->on_floor = 0;
        output->on_wall = 0;
        output->on_ceiling = 0;

        output->floor_normal_x = 0.0;
        output->floor_normal_y = 0.0;
        output->floor_normal_z = 0.0;

        output->wall_normal_x = 0.0;
        output->wall_normal_y = 0.0;
        output->wall_normal_z = 0.0;

        output->ceiling_normal_x = 0.0;
        output->ceiling_normal_y = 0.0;
        output->ceiling_normal_z = 0.0;

        output->last_motion_x = 0.0;
        output->last_motion_y = 0.0;
        output->last_motion_z = 0.0;
        return;
    }

    RID body_rid = rid_from_uint64(input->body_rid_id);

    Ref<PhysicsTestMotionParameters3D> params;
    Ref<PhysicsTestMotionResult3D> res;
    ensure_thread_local_refs(params, res);

    ctx.collision_state = CollisionState();
    ctx.last_motion = Vector3();

    if (ctx.motion_mode == MOTION_MODE_GROUNDED) {
        move_and_slide_grounded(ps, body_rid, ctx, params, res);
    } else {
        move_and_slide_floating(ps, body_rid, ctx, params, res);
    }

    Vector3 real_velocity = (ctx.transform.origin - origin) / ctx.delta;

    output->new_pos_x = ctx.transform.origin.x;
    output->new_pos_y = ctx.transform.origin.y;
    output->new_pos_z = ctx.transform.origin.z;

    output->new_vel_x = ctx.velocity.x;
    output->new_vel_y = ctx.velocity.y;
    output->new_vel_z = ctx.velocity.z;

    output->real_vel_x = real_velocity.x;
    output->real_vel_y = real_velocity.y;
    output->real_vel_z = real_velocity.z;

    output->on_floor = ctx.collision_state.floor ? 1 : 0;
    output->on_wall = ctx.collision_state.wall ? 1 : 0;
    output->on_ceiling = ctx.collision_state.ceiling ? 1 : 0;

    output->floor_normal_x = ctx.floor_normal.x;
    output->floor_normal_y = ctx.floor_normal.y;
    output->floor_normal_z = ctx.floor_normal.z;

    output->wall_normal_x = ctx.wall_normal.x;
    output->wall_normal_y = ctx.wall_normal.y;
    output->wall_normal_z = ctx.wall_normal.z;

    output->ceiling_normal_x = ctx.ceiling_normal.x;
    output->ceiling_normal_y = ctx.ceiling_normal.y;
    output->ceiling_normal_z = ctx.ceiling_normal.z;

    output->last_motion_x = ctx.last_motion.x;
    output->last_motion_y = ctx.last_motion.y;
    output->last_motion_z = ctx.last_motion.z;
}

}
