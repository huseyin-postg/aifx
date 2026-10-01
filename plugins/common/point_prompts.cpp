// Copyright OpenFX and contributors to the OpenFX project.
// SPDX-License-Identifier: BSD-3-Clause

#include "point_prompts.h"
#include "ofxKeySyms.h"

#ifdef __APPLE__
#  ifndef GL_SILENCE_DEPRECATION
#    define GL_SILENCE_DEPRECATION
#  endif
#  include <OpenGL/gl.h>
#else
#  ifdef _WIN32
#    include <windows.h>
#  endif
#  include <GL/gl.h>
#endif

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace ComfyUI {
namespace PointPrompts {

namespace {

// Screen-space sizes, in viewer pixels (converted with DrawArgs/PenArgs::pixelScale).
constexpr double kPointRadius = 6.0;
constexpr double kPickRadius  = 10.0;

void drawPoint(const Point& p, const OfxPointD& ps, bool negative, bool highlighted)
{
    const double r  = kPointRadius * (highlighted ? 1.5 : 1.0);
    const double rx = r * ps.x;
    const double ry = r * ps.y;

    // Dark outline first so the marker reads on bright and dark plates alike.
    for (int pass = 0; pass < 2; ++pass) {
        if (pass == 0) {
            glLineWidth(4.0f);
            glColor3f(0.f, 0.f, 0.f);
        } else {
            glLineWidth(2.0f);
            if (negative) glColor3f(1.f, 0.25f, 0.2f);
            else          glColor3f(0.2f, 1.f, 0.3f);
        }

        glBegin(GL_LINE_LOOP);
        for (int i = 0; i < 24; ++i) {
            const double a = 2.0 * M_PI * i / 24.0;
            glVertex2d(p.x + rx * std::cos(a), p.y + ry * std::sin(a));
        }
        glEnd();

        // '+' for positive, '-' for negative.
        glBegin(GL_LINES);
        glVertex2d(p.x - rx * 0.6, p.y);
        glVertex2d(p.x + rx * 0.6, p.y);
        if (!negative) {
            glVertex2d(p.x, p.y - ry * 0.6);
            glVertex2d(p.x, p.y + ry * 0.6);
        }
        glEnd();
    }
}

std::string nextNodeId(const json& wf)
{
    long maxId = 0;
    for (auto it = wf.begin(); it != wf.end(); ++it) {
        try {
            maxId = std::max(maxId, std::stol(it.key()));
        } catch (...) {
            // Non-numeric keys are allowed by ComfyUI; ignore them.
        }
    }
    return std::to_string(maxId + 1);
}

} // namespace

// ============================================================================
// Parsing / conversion
// ============================================================================

std::vector<Point> parse(const std::string& s)
{
    std::string norm = s;
    std::replace(norm.begin(), norm.end(), ';', ' ');

    std::vector<Point> pts;
    std::istringstream in(norm);
    std::string tok;
    while (in >> tok) {
        const auto comma = tok.find(',');
        if (comma == std::string::npos) continue;
        try {
            Point p{std::stod(tok.substr(0, comma)), std::stod(tok.substr(comma + 1))};
            if (std::isfinite(p.x) && std::isfinite(p.y)) pts.push_back(p);
        } catch (...) {
            // Skip malformed entries rather than failing the whole list.
        }
    }
    return pts;
}

std::string format(const std::vector<Point>& pts)
{
    std::ostringstream out;
    out << std::fixed << std::setprecision(1);
    for (size_t i = 0; i < pts.size(); ++i) {
        if (i) out << ' ';
        out << pts[i].x << ',' << pts[i].y;
    }
    return out.str();
}

std::string toSam3Json(const std::vector<Point>& pts, const OfxRectD& pixelRect,
                       const OfxPointD& scale, bool flipY)
{
    const double w = pixelRect.x2 - pixelRect.x1;
    const double h = pixelRect.y2 - pixelRect.y1;

    json arr = json::array();
    for (const auto& p : pts) {
        double px = p.x * scale.x - pixelRect.x1;
        double py = p.y * scale.y - pixelRect.y1;   // distance from the bottom edge
        if (flipY) py = h - py;                     // EXR rows run top-down
        px = std::clamp(px, 0.0, std::max(0.0, w - 1.0));
        py = std::clamp(py, 0.0, std::max(0.0, h - 1.0));
        arr.push_back({{"x", static_cast<int>(std::lround(px))},
                       {"y", static_cast<int>(std::lround(py))}});
    }
    return arr.dump();
}

void applyToWorkflow(json& wf, Mode mode,
                     const std::string& positiveJson, const std::string& negativeJson,
                     const std::shared_ptr<spdlog::logger>& logger)
{
    if (mode == Mode::Text) return;

    const bool hasPositive = !json::parse(positiveJson).empty();
    const bool hasNegative = !json::parse(negativeJson).empty();

    if (mode == Mode::Points && !hasPositive) {
        throw std::runtime_error(
            "Prompt Mode is 'Points' but no positive point is set. "
            "Click on the subject in the viewer (reference frame) or fill 'Positive Points'.");
    }

    std::string posId, negId;
    if (hasPositive) {
        posId = nextNodeId(wf);
        wf[posId] = {{"inputs", {{"value", positiveJson}}},
                     {"class_type", "PrimitiveString"},
                     {"_meta", {{"title", "AIFX positive points"}}}};
    }
    if (hasNegative) {
        negId = nextNodeId(wf);
        wf[negId] = {{"inputs", {{"value", negativeJson}}},
                     {"class_type", "PrimitiveString"},
                     {"_meta", {{"title", "AIFX negative points"}}}};
    }

    int detectNodes = 0;
    for (auto it = wf.begin(); it != wf.end(); ++it) {
        json& node = it.value();
        if (!node.is_object() || !node.contains("class_type") || !node.contains("inputs")) continue;
        const std::string type = node["class_type"].get<std::string>();
        json& inputs = node["inputs"];

        if (type == "SAM3_Detect") {
            ++detectNodes;
            if (hasPositive) inputs["positive_coords"] = json::array({posId, 0});
            if (hasNegative) inputs["negative_coords"] = json::array({negId, 0});
            if (mode == Mode::Points) inputs.erase("conditioning");
        } else if (type == "SAM3_VideoTrack") {
            // Without this, the tracker keeps detecting new objects from the
            // (possibly empty) text prompt on every detect_interval frame.
            if (mode == Mode::Points) inputs.erase("conditioning");
        }
    }

    if (logger) {
        if (detectNodes == 0) {
            logger->warn("Point prompts set but the workflow has no SAM3_Detect node; points ignored");
        } else {
            logger->info("Point prompts applied to {} SAM3_Detect node(s) (mode={}, positive={}, negative={})",
                         detectNodes, mode == Mode::Points ? "Points" : "Text + Points",
                         positiveJson, negativeJson);
        }
    }
}

// ============================================================================
// Param / overlay description
// ============================================================================

void describeParams(OFX::ImageEffectDescriptor& desc,
                    OFX::PageParamDescriptor* page,
                    OFX::GroupParamDescriptor* parent)
{
    OFX::ChoiceParamDescriptor* mode = desc.defineChoiceParam(kParamPromptMode);
    mode->setLabel("Prompt Mode");
    mode->setHint("Text: segment from the text prompt only.\n"
                  "Points: segment the object under the positive points on the reference frame "
                  "(the text prompt is ignored).\n"
                  "Text + Points: use both.");
    mode->appendOption("Text");
    mode->appendOption("Points");
    mode->appendOption("Text + Points");
    mode->setDefault(static_cast<int>(Mode::Text));
    mode->setAnimates(false);
    if (parent) mode->setParent(*parent);
    if (page) page->addChild(*mode);

    OFX::ChoiceParamDescriptor* clickAdds = desc.defineChoiceParam(kParamClickAdds);
    clickAdds->setLabel("Click Adds");
    clickAdds->setHint("What a click in the viewer adds. Hold Shift while clicking to add the "
                       "opposite kind. Drag a point to move it; hover it and press Delete to remove it.");
    clickAdds->appendOption("Positive Point");
    clickAdds->appendOption("Negative Point");
    clickAdds->setDefault(0);
    clickAdds->setAnimates(false);
    clickAdds->setEvaluateOnChange(false);
    if (parent) clickAdds->setParent(*parent);
    if (page) page->addChild(*clickAdds);

    OFX::StringParamDescriptor* positive = desc.defineStringParam(kParamPositivePoints);
    positive->setLabel("Positive Points");
    positive->setHint("Points on the subject, in canonical image coordinates (origin bottom-left): "
                      "\"x,y x,y ...\". Filled by clicking in the viewer; can also be typed. "
                      "Place them on the Reference Frame.");
    positive->setDefault("");
    positive->setAnimates(false);
    if (parent) positive->setParent(*parent);
    if (page) page->addChild(*positive);

    OFX::StringParamDescriptor* negative = desc.defineStringParam(kParamNegativePoints);
    negative->setLabel("Negative Points");
    negative->setHint("Points to exclude from the mask, same format as Positive Points.");
    negative->setDefault("");
    negative->setAnimates(false);
    if (parent) negative->setParent(*parent);
    if (page) page->addChild(*negative);

    OFX::PushButtonParamDescriptor* undo = desc.definePushButtonParam(kParamUndoPoint);
    undo->setLabel("Undo Last Point");
    undo->setHint("Remove the last point of the kind selected in 'Click Adds'.");
    if (parent) undo->setParent(*parent);
    if (page) page->addChild(*undo);

    OFX::PushButtonParamDescriptor* clear = desc.definePushButtonParam(kParamClearPoints);
    clear->setLabel("Clear Points");
    clear->setHint("Remove all positive and negative points.");
    if (parent) clear->setParent(*parent);
    if (page) page->addChild(*clear);
}

namespace {
class PointOverlayDescriptor
    : public OFX::DefaultEffectOverlayDescriptor<PointOverlayDescriptor, PointOverlay> {};
} // namespace

void describeOverlay(OFX::ImageEffectDescriptor& desc)
{
    desc.setOverlayInteractDescriptor(new PointOverlayDescriptor);
}

// ============================================================================
// PointOverlay
// ============================================================================

PointOverlay::PointOverlay(OfxInteractHandle handle, OFX::ImageEffect* effect)
    : OFX::OverlayInteract(handle)
{
    if (!effect) return;
    if (effect->paramExists(kParamPromptMode))     _mode      = effect->fetchChoiceParam(kParamPromptMode);
    if (effect->paramExists(kParamPositivePoints)) _positive  = effect->fetchStringParam(kParamPositivePoints);
    if (effect->paramExists(kParamNegativePoints)) _negative  = effect->fetchStringParam(kParamNegativePoints);
    if (effect->paramExists(kParamClickAdds))      _clickAdds = effect->fetchChoiceParam(kParamClickAdds);

    // Redraw the overlay whenever the points or the mode change from the panel.
    if (_mode)     addParamToSlaveTo(_mode);
    if (_positive) addParamToSlaveTo(_positive);
    if (_negative) addParamToSlaveTo(_negative);
}

bool PointOverlay::active() const
{
    if (!_mode || !_positive || !_negative) return false;
    int m = 0;
    _mode->getValue(m);
    return m != static_cast<int>(Mode::Text);
}

std::vector<Point> PointOverlay::load(bool negative) const
{
    std::string s;
    (negative ? _negative : _positive)->getValue(s);
    return parse(s);
}

void PointOverlay::store(bool negative, const std::vector<Point>& pts)
{
    (negative ? _negative : _positive)->setValue(format(pts));
}

bool PointOverlay::pick(const OfxPointD& pos, const OfxPointD& ps, int& index, bool& negative) const
{
    double best = kPickRadius;
    bool found = false;
    for (int list = 0; list < 2; ++list) {
        const auto pts = load(list == 1);
        for (size_t i = 0; i < pts.size(); ++i) {
            const double dx = (pts[i].x - pos.x) / (ps.x > 0 ? ps.x : 1.0);
            const double dy = (pts[i].y - pos.y) / (ps.y > 0 ? ps.y : 1.0);
            const double d = std::sqrt(dx * dx + dy * dy);
            if (d <= best) {
                best = d;
                index = static_cast<int>(i);
                negative = (list == 1);
                found = true;
            }
        }
    }
    return found;
}

bool PointOverlay::draw(const OFX::DrawArgs& args)
{
    if (!active()) return false;
    for (int list = 0; list < 2; ++list) {
        const bool neg = (list == 1);
        const auto pts = load(neg);
        for (size_t i = 0; i < pts.size(); ++i) {
            const bool hl = (static_cast<int>(i) == _hoverIndex && neg == _hoverNegative) ||
                            (static_cast<int>(i) == _dragIndex && neg == _dragNegative);
            drawPoint(pts[i], args.pixelScale, neg, hl);
        }
    }
    return true;
}

bool PointOverlay::penDown(const OFX::PenArgs& args)
{
    if (!active()) return false;

    int idx = -1;
    bool neg = false;
    if (pick(args.penPosition, args.pixelScale, idx, neg)) {
        _dragIndex = idx;
        _dragNegative = neg;
        _dragMoved = false;
        return true;
    }

    int clickAdds = 0;
    if (_clickAdds) _clickAdds->getValue(clickAdds);
    const bool addNegative = (clickAdds == 1) != _shiftDown;

    auto pts = load(addNegative);
    pts.push_back({args.penPosition.x, args.penPosition.y});
    _effect->beginEditBlock("Add point");
    store(addNegative, pts);
    _effect->endEditBlock();

    // Keep the new point under the pen so press-and-drag places it precisely.
    _dragIndex = static_cast<int>(pts.size()) - 1;
    _dragNegative = addNegative;
    _dragMoved = false;
    requestRedraw();
    return true;
}

bool PointOverlay::penMotion(const OFX::PenArgs& args)
{
    if (!active()) return false;

    if (_dragIndex >= 0) {
        auto pts = load(_dragNegative);
        if (_dragIndex < static_cast<int>(pts.size())) {
            pts[_dragIndex] = {args.penPosition.x, args.penPosition.y};
            store(_dragNegative, pts);
            _dragMoved = true;
        }
        requestRedraw();
        return true;
    }

    int idx = -1;
    bool neg = false;
    const bool hit = pick(args.penPosition, args.pixelScale, idx, neg);
    const int newHover = hit ? idx : -1;
    if (newHover != _hoverIndex || (hit && neg != _hoverNegative)) {
        _hoverIndex = newHover;
        _hoverNegative = neg;
        requestRedraw();
    }
    return false;
}

bool PointOverlay::penUp(const OFX::PenArgs& /*args*/)
{
    if (_dragIndex < 0) return false;
    _dragIndex = -1;
    _dragMoved = false;
    requestRedraw();
    return true;
}

bool PointOverlay::keyDown(const OFX::KeyArgs& args)
{
    if (args.keySymbol == kOfxKey_Shift_L || args.keySymbol == kOfxKey_Shift_R) {
        _shiftDown = true;
        return false;   // never swallow modifiers
    }
    if (!active()) return false;
    if ((args.keySymbol == kOfxKey_Delete || args.keySymbol == kOfxKey_BackSpace) && _hoverIndex >= 0) {
        auto pts = load(_hoverNegative);
        if (_hoverIndex < static_cast<int>(pts.size())) {
            pts.erase(pts.begin() + _hoverIndex);
            _effect->beginEditBlock("Remove point");
            store(_hoverNegative, pts);
            _effect->endEditBlock();
        }
        _hoverIndex = -1;
        requestRedraw();
        return true;
    }
    return false;
}

bool PointOverlay::keyUp(const OFX::KeyArgs& args)
{
    if (args.keySymbol == kOfxKey_Shift_L || args.keySymbol == kOfxKey_Shift_R) {
        _shiftDown = false;
    }
    return false;
}

} // namespace PointPrompts
} // namespace ComfyUI
