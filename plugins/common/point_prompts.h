// Copyright OpenFX and contributors to the OpenFX project.
// SPDX-License-Identifier: BSD-3-Clause

#ifndef COMFYUI_POINT_PROMPTS_H
#define COMFYUI_POINT_PROMPTS_H

/**
 * @file point_prompts.h
 * @brief Click/point prompts for the SAM3-based plugins (SAM3, MaMa, MatAnyone2).
 *
 * The artist places positive / negative points on the reference frame, either by
 * clicking in the host viewer (overlay interact) or by typing them into the
 * "Positive Points" / "Negative Points" string params. Points are stored in OFX
 * canonical coordinates (origin bottom-left, x scaled by the pixel aspect ratio)
 * as "x,y x,y ..." so they survive project save/load in every host.
 *
 * At submit time the points are converted to the pixel space of the EXRs the
 * plugin writes (origin top-left when the plugin Y-flips for the host) and wired
 * into every SAM3_Detect node of the workflow as positive_coords /
 * negative_coords. In "Points" mode the text conditioning is unplugged from
 * SAM3_Detect and SAM3_VideoTrack so an empty / stale prompt cannot add objects.
 */

#include "ofxsImageEffect.h"
#include "ofxsInteract.h"
#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>
#include <memory>
#include <string>
#include <vector>

namespace ComfyUI {
namespace PointPrompts {

using json = nlohmann::json;

// Param names (shared by every plugin that calls describeParams()).
inline constexpr const char* kParamPromptMode     = "promptMode";
inline constexpr const char* kParamPositivePoints = "positivePoints";
inline constexpr const char* kParamNegativePoints = "negativePoints";
inline constexpr const char* kParamClickAdds      = "pointClickAdds";
inline constexpr const char* kParamUndoPoint      = "undoLastPoint";
inline constexpr const char* kParamClearPoints    = "clearPoints";

// Order matches the promptMode choice options.
enum class Mode { Text = 0, Points = 1, TextAndPoints = 2 };

struct Point { double x; double y; };

/** Parse "x,y x,y" (separators: whitespace or ';'). Malformed entries are skipped. */
std::vector<Point> parse(const std::string& s);

/** Inverse of parse(); one decimal place is plenty for canonical pixel coords. */
std::string format(const std::vector<Point>& pts);

/**
 * Convert canonical points to the SAM3_Detect JSON format
 * [{"x": int, "y": int}, ...] in the pixel space of the written EXRs.
 *
 * @param pixelRect Bounds of the written image in OFX pixel coords (bottom-left origin).
 * @param scale     Canonical -> pixel factor: (renderScale.x / PAR, renderScale.y).
 * @param flipY     true when the plugin flips rows for EXR (host origin bottom-left).
 */
std::string toSam3Json(const std::vector<Point>& pts, const OfxRectD& pixelRect,
                       const OfxPointD& scale, bool flipY);

/**
 * Wire point prompts into every SAM3_Detect node of @p wf and, in Points mode,
 * unplug the text conditioning from SAM3_Detect / SAM3_VideoTrack.
 * No-op in Text mode. Throws std::runtime_error when Points mode has no positive
 * point (SAM3_VideoTrack would otherwise get neither a mask nor a prompt).
 */
void applyToWorkflow(json& wf, Mode mode,
                     const std::string& positiveJson, const std::string& negativeJson,
                     const std::shared_ptr<spdlog::logger>& logger);

/** Define the point-prompt params under @p parent (call from describeInContext). */
void describeParams(OFX::ImageEffectDescriptor& desc,
                    OFX::PageParamDescriptor* page,
                    OFX::GroupParamDescriptor* parent);

/** Register the viewer overlay that lets the artist click points (call from describe). */
void describeOverlay(OFX::ImageEffectDescriptor& desc);

/** Viewer overlay: click to add, drag to move, Delete/Backspace removes the hovered point. */
class PointOverlay : public OFX::OverlayInteract {
public:
    PointOverlay(OfxInteractHandle handle, OFX::ImageEffect* effect);

    bool draw(const OFX::DrawArgs& args) override;
    bool penDown(const OFX::PenArgs& args) override;
    bool penMotion(const OFX::PenArgs& args) override;
    bool penUp(const OFX::PenArgs& args) override;
    bool keyDown(const OFX::KeyArgs& args) override;
    bool keyUp(const OFX::KeyArgs& args) override;

private:
    OFX::ChoiceParam* _mode     = nullptr;
    OFX::StringParam* _positive = nullptr;
    OFX::StringParam* _negative = nullptr;
    OFX::ChoiceParam* _clickAdds = nullptr;

    // Drag / hover state. _dragNegative selects which list _dragIndex refers to.
    int  _dragIndex    = -1;
    bool _dragNegative = false;
    bool _dragMoved    = false;
    int  _hoverIndex   = -1;
    bool _hoverNegative = false;
    bool _shiftDown    = false;

    bool active() const;
    // Finds the point nearest to @p pos within the pick radius; returns false if none.
    bool pick(const OfxPointD& pos, const OfxPointD& pixelScale, int& index, bool& negative) const;
    std::vector<Point> load(bool negative) const;
    void store(bool negative, const std::vector<Point>& pts);
};

} // namespace PointPrompts
} // namespace ComfyUI

#endif // COMFYUI_POINT_PROMPTS_H
