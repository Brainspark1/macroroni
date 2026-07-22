# Passing Actions - Step by Step Implementation Plan

## Overview

Goal: When a user says "jump" (action-only, no target), NESBERT extracts the ACTION entity, which goes through SemanticMapper to resolve the canonical action name, then mapped to emulator buttons via the JSON `button` field.

---

## Step 1: Fix SemanticMapper.find_max_action_similarity bug

**File:** `passing_actions/macroroni/library/SemanticMapper.py` (line ~62)
**Change:** `target_names` → `action_names`

---

## Step 2: MarioVoiceController.process_game_commands → route actions through SemanticMapper

**File:** `passing_actions/macroroni/library/MarioVoiceController.py` (lines 27-36)
**Change:** When action words found alone, call `set_action_from_similarity()` instead of storing raw word.

---

## Step 3: Fix AutoEnemyTracking.deactivate typo

**File:** `passing_actions/macroroni/library/AutoEnemyTracking.py`
**Change:** `self.action_type_name = Non` → `self.action_type_name = None`

---

## Step 4: main.py emulator loop — handle action button presses

**File:** `passing_actions/macroroni/library/main.py`
**Change:** Replace old `pass_movement` block with `auto_enemy_tracking.passing_action` + button lookup from JSON.

---

## Step 5: ✅ Already done — json_file.json has expanded descriptions + button mapping

---

## Step 6: Fix mapping_json_path in main.py

**File:** `passing_actions/macroroni/library/main.py` (line ~53)
**Bug:** Still points to `connecting_library/...` (old JSON without `actions` key).
**Change:** Switch to `passing_actions/macroroni/library/json_file.json`

---

## Step 7: Add action_hold_frames to AutoEnemyTracking

**File:** `passing_actions/macroroni/library/AutoEnemyTracking.py`
**Why:** `getattr(auto_enemy_tracking, "action_hold_frames", 0)` returns 0 → action resets after 1 frame.

- Add `self.action_hold_frames = 0` and `self.action_hold_durations` dict in `__init__`
- Set `self.action_hold_frames = self.action_hold_durations.get(name, 10)` in `set_action_from_similarity`
- Reset to 0 in both `activate()` and `deactivate()`

---

## Step 8: Remove stale `voice_controller.pass_movement` in main.py

**File:** `passing_actions/macroroni/library/main.py`
**Change:** Remove `voice_controller.pass_movement = False` line (no longer needed).

---

## 🆕 Step 9: Fix auto tracking getting stuck — Mario turns toward enemy but doesn't approach

**File:** `passing_actions/macroroni/library/AutoEnemyTracking.py`

### Root Cause Analysis:

Three bugs combine to freeze Mario:

### 🐛 Bug 9a: Candidate filter kills ALL movement when no enemies match target_type

**File:** `passing_actions/macroroni/library/AutoEnemyTracking.py` (~lines 213-218)

When `self.target_type_address` is set (e.g. `0x06` for goomba):

- `candidates` filters to only goomba-type enemies
- If no goombas in the current enemy list (e.g. they died or left screen), `candidates` is empty
- `get_action` returns `["NOOP"]` → Mario stands still FOREVER

**Fix:** Fall back to ALL enemies if no matching type found, rather than returning NOOP.

```python
# Replace this block:
if self.target_type_address is not None:
    candidates = [
        enemy
        for enemy in enemy_profiles
        if enemy.get("enemy_type") == self.target_type_address
    ]
else:
    candidates = enemy_profiles

if not candidates:
    return COMPLEX_MOVEMENT.index(["NOOP"])

# With:
if self.target_type_address is not None:
    candidates = [
        enemy
        for enemy in enemy_profiles
        if enemy.get("enemy_type") == self.target_type_address
    ]
    # Fall back to ALL enemies if no matching type found
    if not candidates:
        candidates = enemy_profiles
else:
    candidates = enemy_profiles

if not candidates:
    self.deactivate()
    return COMPLEX_MOVEMENT.index(["NOOP"])
```

### 🐛 Bug 9b: max_x_distance (15) and max_kill_x (22) are too small

Super Mario Bros sprites are ~16x16 pixels. Killing distance of 22 means Mario must be almost touching the goomba. With terrain bumps and slight elevation differences, Mario wiggles at the edge forever.

**Fix:** Increase thresholds for more aggressive pursuit.

```python
# In __init__ defaults, change:
# Old: max_x_distance=15, max_kill_x=22
# New:
max_x_distance=40,
max_kill_x=50,
frames_decide_run=30,
```

### 🐛 Bug 9c: Final approach uses walk (no B button), too slow to reach kill range

The last block in `get_action`:

```python
# OLD (lines ~253-256):
if horizontal_distance >= 0:
    return COMPLEX_MOVEMENT.index(["right"])
else:
    return COMPLEX_MOVEMENT.index(["left"])
```

Mario crawls at walking speed and may never close the gap enough within the kill threshold.

**Fix:** Use run button (B) for final approach too:

```python
# NEW:
if horizontal_distance >= 0:
    return COMPLEX_MOVEMENT.index(["right", "B"])
else:
    return COMPLEX_MOVEMENT.index(["left", "B"])
```

### 📝 Summary of changes to AutoEnemyTracking.py:

1. `__init__` params: change max_x_distance=40, max_kill_x=50, frames_decide_run=30
2. `get_action()`: Change candidates fallback from NOOP → use all enemies
3. `get_action()`: Change final approach from walk → run (add "B" button)
