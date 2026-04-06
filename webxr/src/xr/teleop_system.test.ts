import { Group, Matrix4, Quaternion, Vector3 } from "three";
import { beforeAll, describe, expect, it, vi } from "vitest";

type TeleopSystemType = import("./teleop_system").TeleopSystem;

let TeleopSystem: typeof import("./teleop_system").TeleopSystem;

// ---------------------------------------------------------------------------
// DOM stub – `@iwsdk/core` calls `document.createElement("canvas")` at import
// time, so we must provide a minimal polyfill before the dynamic import.
// ---------------------------------------------------------------------------
vi.stubGlobal("document", {
	createElement: (tag: string) => {
		if (tag === "canvas") {
			return {
				width: 0,
				height: 0,
				getContext: () => ({
					clearRect() {},
					beginPath() {},
					arc() {},
					fill() {},
					stroke() {},
					fillStyle: "white",
					strokeStyle: "gray",
					lineWidth: 1,
				}),
			};
		}
		return {};
	},
	body: {},
});

/**
 * Create a lightweight stub of TeleopSystem that shares the real prototype
 * methods (e.g. `buildHandDevice`, `gatherInputState`) but skips the
 * constructor, which requires a live ECS world, WebSocket, etc.
 *
 * The `temp*` fields are working buffers used by the production code for
 * matrix decomposition; they must be initialised manually because we bypass
 * the constructor.
 */
function createTeleopSystemStub() {
	const system = Object.create(TeleopSystem.prototype) as {
		tempPosition: Vector3;
		tempQuaternion: Quaternion;
		tempMatrix: Matrix4;
		tempScale: Vector3;
		world: unknown;
		buildHandDevice: TeleopSystemType["buildHandDevice"];
		gatherInputState: TeleopSystemType["gatherInputState"];
	};
	system.tempPosition = new Vector3();
	system.tempQuaternion = new Quaternion();
	system.tempMatrix = new Matrix4();
	system.tempScale = new Vector3();
	return system;
}

describe("TeleopSystem", () => {
	beforeAll(async () => {
		({ TeleopSystem } = await import("./teleop_system"));
	});

	// -----------------------------------------------------------------------
	// buildHandDevice
	// -----------------------------------------------------------------------
	describe("buildHandDevice", () => {
		it("returns hand device with world-space joint poses when adapter is connected", () => {
			const system = createTeleopSystemStub();
			const gripSpace = new Group();
			gripSpace.position.set(1, 2, 3);
			gripSpace.quaternion.setFromAxisAngle(
				new Vector3(0, 0, 1),
				Math.PI / 2,
			);
			gripSpace.updateMatrixWorld(true);

			const localJointMatrix = new Matrix4().makeTranslation(0.1, 0, 0);
			const handDevice = system.buildHandDevice("left", {
				connected: true,
				gripSpace,
				jointSpaces: [{ jointName: "wrist" }],
				jointTransforms: new Float32Array(localJointMatrix.toArray()),
			});

			expect(handDevice).not.toBeNull();
			expect(handDevice?.role).toBe("hand");
			expect(handDevice?.handedness).toBe("left");
			expect(handDevice?.joints?.wrist.position.x).toBeCloseTo(1, 6);
			expect(handDevice?.joints?.wrist.position.y).toBeCloseTo(2.1, 6);
			expect(handDevice?.joints?.wrist.position.z).toBeCloseTo(3, 6);
			expect(handDevice?.joints?.wrist.orientation.x).toBeCloseTo(0, 6);
			expect(handDevice?.joints?.wrist.orientation.y).toBeCloseTo(0, 6);
			expect(handDevice?.joints?.wrist.orientation.z).toBeCloseTo(
				Math.sqrt(0.5),
				6,
			);
			expect(handDevice?.joints?.wrist.orientation.w).toBeCloseTo(
				Math.sqrt(0.5),
				6,
			);
		});

		it("returns null when adapter is not connected", () => {
			const system = createTeleopSystemStub();
			const gripSpace = new Group();
			gripSpace.updateMatrixWorld(true);

			const result = system.buildHandDevice("right", {
				connected: false,
				gripSpace,
				jointSpaces: [{ jointName: "wrist" }],
				jointTransforms: new Float32Array(
					new Matrix4().identity().toArray(),
				),
			});

			expect(result).toBeNull();
		});

		it("returns null when adapter is null", () => {
			const system = createTeleopSystemStub();
			const result = system.buildHandDevice("left", null);
			expect(result).toBeNull();
		});

		it("returns null when adapter is undefined", () => {
			const system = createTeleopSystemStub();
			const result = system.buildHandDevice("right", undefined);
			expect(result).toBeNull();
		});

		it("returns device with empty joints when jointSpaces is empty", () => {
			const system = createTeleopSystemStub();
			const gripSpace = new Group();
			gripSpace.position.set(1, 0, 0);
			gripSpace.updateMatrixWorld(true);

			const handDevice = system.buildHandDevice("left", {
				connected: true,
				gripSpace,
				jointSpaces: [],
				jointTransforms: new Float32Array([]),
			});

			expect(handDevice).not.toBeNull();
			expect(handDevice?.role).toBe("hand");
			expect(handDevice?.joints).toEqual({});
		});
	});

	// -----------------------------------------------------------------------
	// gatherInputState – hand adapters
	// -----------------------------------------------------------------------
	describe("gatherInputState", () => {
		function createWorldWithHead(headPos: [number, number, number]) {
			const head = new Group();
			head.position.set(...headPos);
			head.updateMatrixWorld(true);
			return { player: { head, raySpaces: {} } };
		}

		const defaultSettings = {
			speed: 1,
			turnSpeed: 1,
			precisionMode: false,
		};

		it("includes a connected hand adapter in the device list", () => {
			const system = createTeleopSystemStub();
			system.world = createWorldWithHead([0, 0, 1]);

			const gripSpace = new Group();
			gripSpace.position.set(0.5, 0.25, 0.75);
			gripSpace.updateMatrixWorld(true);

			const state = system.gatherInputState(
				{
					visualAdapters: {
						hand: {
							left: {
								connected: true,
								gripSpace,
								jointSpaces: [{ jointName: "wrist" }],
								jointTransforms: new Float32Array(
									new Matrix4().identity().toArray(),
								),
							},
						},
					},
				},
				defaultSettings,
			);

			expect(state).not.toBeNull();
			expect(state?.devices.map((d) => d.role)).toEqual(["head", "hand"]);
			expect(state?.devices[1]?.handedness).toBe("left");
			expect(state?.devices[1]?.joints?.wrist.position.x).toBeCloseTo(
				0.5,
				6,
			);
			expect(state?.devices[1]?.joints?.wrist.position.y).toBeCloseTo(
				0.25,
				6,
			);
			expect(state?.devices[1]?.joints?.wrist.position.z).toBeCloseTo(
				0.75,
				6,
			);
		});

		it("includes both left and right hand devices when both adapters are connected", () => {
			const system = createTeleopSystemStub();
			system.world = createWorldWithHead([0, 1, 0]);

			const leftGrip = new Group();
			leftGrip.position.set(-0.3, 1, 0.5);
			leftGrip.updateMatrixWorld(true);

			const rightGrip = new Group();
			rightGrip.position.set(0.3, 1, 0.5);
			rightGrip.updateMatrixWorld(true);

			const identity = new Float32Array(
				new Matrix4().identity().toArray(),
			);

			const state = system.gatherInputState(
				{
					visualAdapters: {
						hand: {
							left: {
								connected: true,
								gripSpace: leftGrip,
								jointSpaces: [{ jointName: "wrist" }],
								jointTransforms: identity,
							},
							right: {
								connected: true,
								gripSpace: rightGrip,
								jointSpaces: [{ jointName: "wrist" }],
								jointTransforms: identity,
							},
						},
					},
				},
				defaultSettings,
			);

			expect(state).not.toBeNull();
			const roles = state?.devices.map((d) => d.role);
			expect(roles).toEqual(["head", "hand", "hand"]);
			const handDevices = state?.devices.filter(
				(d) => d.role === "hand",
			);
			expect(handDevices?.map((d) => d.handedness)).toEqual([
				"left",
				"right",
			]);
		});
	});
});
