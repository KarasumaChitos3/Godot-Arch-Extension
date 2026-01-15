using System;
using System.Runtime.InteropServices;
using Godot;

public partial class MyPhysicsExample : Node
{
    [StructLayout(LayoutKind.Sequential, Pack = 1)]
    public struct MoveParams
    {
        public float t_basis_xx, t_basis_xy, t_basis_xz;
        public float t_basis_yx, t_basis_yy, t_basis_yz;
        public float t_basis_zx, t_basis_zy, t_basis_zz;
        public float t_origin_x, t_origin_y, t_origin_z;

        public float vel_x, vel_y, vel_z;
        public float up_x, up_y, up_z;

        public float delta;
        public float floor_max_angle;
        public float floor_snap_length;
        public float safe_margin;
        public float wall_min_slide_angle;

        public int max_slides;

        public byte motion_mode; // 0=grounded, 1=floating
        public byte floor_stop_on_slope;
        public byte floor_constant_speed;
        public byte floor_block_on_wall;
        public byte slide_on_ceiling;
        public byte was_on_floor;
        public byte locked_axis; // bitmask: x=1, y=2, z=4
        public byte reserved0;

        public float prev_floor_normal_x, prev_floor_normal_y, prev_floor_normal_z;

        public ulong body_rid_id;
    }

    [StructLayout(LayoutKind.Sequential, Pack = 1)]
    public struct MoveResult
    {
        public float new_pos_x, new_pos_y, new_pos_z;
        public float new_vel_x, new_vel_y, new_vel_z;
        public float real_vel_x, real_vel_y, real_vel_z;

        public byte on_floor;
        public byte on_wall;
        public byte on_ceiling;
        public byte reserved0;

        public float floor_normal_x, floor_normal_y, floor_normal_z;
        public float wall_normal_x, wall_normal_y, wall_normal_z;
        public float ceiling_normal_x, ceiling_normal_y, ceiling_normal_z;

        public float last_motion_x, last_motion_y, last_motion_z;
    }

    private static class Native
    {
        private const string LibName = "Godot-ArchExtension";

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl, EntryPoint = "c_move_and_slide")]
        public static extern void MoveAndSlide(ref MoveParams input, out MoveResult output);
    }

    [Export] public CollisionObject3D Target;
    [Export] public bool ApplyResultToTarget = true;

    private Vector3 _velocity = Vector3.Zero;
    private bool _wasOnFloor;
    private Vector3 _prevFloorNormal = Vector3.Up;

    public override void _PhysicsProcess(double delta)
    {
        if (Target == null)
        {
            return;
        }

        Transform3D t = Target.GlobalTransform;
        Basis b = t.Basis;
        Vector3 origin = t.Origin;

        MoveParams input = default;
        input.t_basis_xx = b.X.X;
        input.t_basis_xy = b.X.Y;
        input.t_basis_xz = b.X.Z;
        input.t_basis_yx = b.Y.X;
        input.t_basis_yy = b.Y.Y;
        input.t_basis_yz = b.Y.Z;
        input.t_basis_zx = b.Z.X;
        input.t_basis_zy = b.Z.Y;
        input.t_basis_zz = b.Z.Z;
        input.t_origin_x = origin.X;
        input.t_origin_y = origin.Y;
        input.t_origin_z = origin.Z;

        input.vel_x = _velocity.X;
        input.vel_y = _velocity.Y;
        input.vel_z = _velocity.Z;

        input.up_x = 0.0f;
        input.up_y = 1.0f;
        input.up_z = 0.0f;

        input.delta = (float)delta;
        input.floor_max_angle = Mathf.DegToRad(45.0f);
        input.floor_snap_length = 0.1f;
        input.safe_margin = 0.001f;
        input.wall_min_slide_angle = 0.0f;

        input.max_slides = 6;

        input.motion_mode = 0;
        input.floor_stop_on_slope = 0;
        input.floor_constant_speed = 0;
        input.floor_block_on_wall = 0;
        input.slide_on_ceiling = 0;
        input.was_on_floor = _wasOnFloor ? (byte)1 : (byte)0;
        input.locked_axis = 0;
        input.reserved0 = 0;

        input.prev_floor_normal_x = _prevFloorNormal.X;
        input.prev_floor_normal_y = _prevFloorNormal.Y;
        input.prev_floor_normal_z = _prevFloorNormal.Z;

        input.body_rid_id = Target.GetRid().Id;

        Native.MoveAndSlide(ref input, out MoveResult result);

        _velocity = new Vector3(result.new_vel_x, result.new_vel_y, result.new_vel_z);
        _wasOnFloor = result.on_floor != 0;
        _prevFloorNormal = new Vector3(result.floor_normal_x, result.floor_normal_y, result.floor_normal_z);

        if (ApplyResultToTarget)
        {
            Transform3D newTransform = new Transform3D(t.Basis, new Vector3(result.new_pos_x, result.new_pos_y, result.new_pos_z));
            Target.GlobalTransform = newTransform;
        }
    }
}
