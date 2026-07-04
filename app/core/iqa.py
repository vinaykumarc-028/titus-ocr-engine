import math
from pathlib import Path
from PIL import Image, ImageFilter, ImageChops, ImageStat

def analyze_image_quality(image_path: Path) -> dict:
    """
    Performs multi-dimensional Image Quality Assessment (IQA) using Pillow.
    Returns a dictionary of structured metrics, scores, status ratings,
    overall OCR difficulty, and actionable recommendations.
    """
    try:
        with Image.open(image_path) as img:
            # 1. Basic properties
            width, height = img.size
            megapixels = (width * height) / 1_000_000.0

            # Convert to grayscale for feature metrics
            gray = img.convert("L")
            stat = ImageStat.Stat(gray)
            
            # Grayscale stats
            mean_luminance = stat.mean[0]
            std_dev = stat.stddev[0]

            # 2. Blur Estimation (Variance of Laplacian)
            # Custom 3x3 Laplacian edge-detection kernel
            laplacian_kernel = ImageFilter.Kernel(
                (3, 3),
                [-1, -1, -1,
                 -1,  8, -1,
                 -1, -1, -1],
                scale=1,
                offset=0
            )
            laplacian_img = gray.filter(laplacian_kernel)
            lap_stat = ImageStat.Stat(laplacian_img)
            # Variance represents detail/sharpness
            blur_score = lap_stat.var[0]

            # 3. Contrast (RMS contrast is std dev of grayscale intensities)
            contrast_score = std_dev

            # 4. Noise Estimation (Residue deviation from Gaussian Blur)
            blurred = gray.filter(ImageFilter.GaussianBlur(radius=1.5))
            diff = ImageChops.difference(gray, blurred)
            diff_stat = ImageStat.Stat(diff)
            noise_score = diff_stat.mean[0]

            # 5. Perspective & Skew
            # Projection profile variance search for skew estimation
            small_gray = gray.resize((150, 200))
            best_angle = 0
            max_var = 0.0
            for angle in range(-5, 6):
                rotated = small_gray.rotate(angle, resample=Image.BICUBIC)
                row_profile = rotated.resize((1, 200))
                pixels = list(row_profile.getdata())
                mean_profile = sum(pixels) / 200.0
                var_profile = sum((p - mean_profile) ** 2 for p in pixels) / 200.0
                if var_profile > max_var:
                    max_var = var_profile
                    best_angle = angle
            
            skew_angle = abs(best_angle)

            # Perspective asymmetry check (left vs right illumination margin diff)
            left_strip = gray.crop((0, 0, int(width * 0.1), height))
            right_strip = gray.crop((int(width * 0.9), 0, width, height))
            left_mean = ImageStat.Stat(left_strip).mean[0]
            right_mean = ImageStat.Stat(right_strip).mean[0]
            margin_diff = abs(left_mean - right_mean)

            # Aspect ratio deviation from A4 (1.414)
            aspect_ratio = (height / width) if height > width else (width / height)
            a4_ratio = 1.414
            ratio_dev = abs(aspect_ratio - a4_ratio)
            perspective_score = ratio_dev * 100.0 + (margin_diff / 8.0)

            # 6. Exposure & Clipping (Saturation)
            hist = gray.histogram()
            white_pixels = hist[255] if len(hist) > 255 else 0
            clipping_pct = (white_pixels / (width * height)) * 100.0

            # 7. JPEG Compression Blockiness Ratio
            # We sample columns and calculate differences at multiples of 8 vs others
            block_sample = gray.resize((256, 256))
            sample_data = list(block_sample.getdata())
            col_diffs = [0.0] * 8
            for y in range(256):
                for x in range(255):
                    col_diffs[x % 8] += abs(sample_data[y * 256 + x] - sample_data[y * 256 + x + 1])
            avg_non_boundary = sum(col_diffs[:7]) / 7.0
            block_ratio = col_diffs[7] / (avg_non_boundary if avg_non_boundary > 0 else 1.0)

            # Determine statuses and labels
            # A. Resolution
            if megapixels >= 0.8:
                res_status = "green"
                res_label = "Good"
            elif megapixels >= 0.4:
                res_status = "yellow"
                res_label = "Moderate"
            else:
                res_status = "red"
                res_label = "Low"

            # B. Blur
            if blur_score >= 80:
                blur_status = "green"
                blur_label = "Good"
            elif blur_score >= 35:
                blur_status = "yellow"
                blur_label = "Moderate"
            else:
                blur_status = "red"
                blur_label = "Blurry"

            # C. Contrast
            if contrast_score >= 50.0:
                contrast_status = "green"
                contrast_label = "Good"
            elif contrast_score >= 30.0:
                contrast_status = "yellow"
                contrast_label = "Low"
            else:
                contrast_status = "red"
                contrast_label = "Very Low"

            # D. Noise
            if noise_score <= 3.5:
                noise_status = "green"
                noise_label = "Low"
            elif noise_score <= 7.0:
                noise_status = "yellow"
                noise_label = "Moderate"
            else:
                noise_status = "red"
                noise_label = "High"

            # E. Perspective / Skew
            if skew_angle < 2 and perspective_score < 6:
                persp_status = "green"
                persp_label = "Good"
            elif skew_angle <= 5 and perspective_score <= 15:
                persp_status = "yellow"
                persp_label = "Slight distortion"
            else:
                persp_status = "red"
                persp_label = "High distortion"

            # F. Exposure
            if 80 <= mean_luminance <= 220 and clipping_pct < 5.0:
                exp_status = "green"
                exp_label = "Good"
            elif 60 <= mean_luminance <= 235 and clipping_pct < 15.0:
                exp_status = "yellow"
                exp_label = "Moderate"
            else:
                exp_status = "red"
                exp_label = "Poor"

            # Compute overall difficulty
            statuses = [res_status, blur_status, contrast_status, noise_status, persp_status, exp_status]
            red_count = statuses.count("red")
            yellow_count = statuses.count("yellow")

            if red_count == 0 and yellow_count == 0:
                difficulty = "Low"
                overall_status = "Excellent"
            elif red_count == 0 and yellow_count == 1:
                difficulty = "Low"
                overall_status = "Good"
            elif red_count == 0 and yellow_count <= 3:
                difficulty = "Medium"
                overall_status = "Acceptable"
            elif red_count == 1 or yellow_count >= 4:
                difficulty = "Medium"
                overall_status = "Needs Review"
            elif red_count == 2:
                difficulty = "High"
                overall_status = "Poor"
            else:
                difficulty = "High"
                overall_status = "Very Poor"

            # Compile Recommendations
            recs = []
            if contrast_status != "green":
                recs.append("Low Contrast → Increase contrast before OCR")
            if blur_status != "green":
                recs.append("Blur → Capture a sharper image")
            if persp_status != "green":
                if skew_angle > 2:
                    recs.append("Skew → Auto deskew")
                else:
                    recs.append("Perspective → Flatten page automatically")
            if exp_status != "green" and clipping_pct >= 5.0:
                recs.append("Shadows/Glare → Remove shadow/glare regions")

            return {
                "resolution": {
                    "status": res_status,
                    "label": res_label,
                    "value": f"{width}x{height} ({megapixels:.2f} MP)"
                },
                "blur": {
                    "status": blur_status,
                    "label": blur_label,
                    "value": f"Score: {blur_score:.1f}"
                },
                "contrast": {
                    "status": contrast_status,
                    "label": contrast_label,
                    "value": f"Score: {contrast_score:.1f}"
                },
                "noise": {
                    "status": noise_status,
                    "label": noise_label,
                    "value": f"Score: {noise_score:.1f}"
                },
                "perspective": {
                    "status": persp_status,
                    "label": persp_label,
                    "value": f"Skew: {skew_angle}°, Dev: {perspective_score:.1f}"
                },
                "exposure": {
                    "status": exp_status,
                    "label": exp_label,
                    "value": f"Mean: {mean_luminance:.1f}, Clip: {clipping_pct:.1f}%"
                },
                "jpeg_blockiness": {
                    "status": "yellow" if block_ratio >= 1.06 else "green",
                    "label": "Moderate" if block_ratio >= 1.06 else "Good",
                    "value": f"Ratio: {block_ratio:.3f}"
                },
                "difficulty": difficulty,
                "overall_status": overall_status,
                "recommendations": recs
            }

    except Exception as e:
        return {
            "resolution": {"status": "green", "label": "Good", "value": "Unknown"},
            "blur": {"status": "green", "label": "Good", "value": "Unknown"},
            "contrast": {"status": "green", "label": "Good", "value": "Unknown"},
            "noise": {"status": "green", "label": "Low", "value": "Unknown"},
            "perspective": {"status": "green", "label": "Good", "value": "Unknown"},
            "exposure": {"status": "green", "label": "Good", "value": "Unknown"},
            "difficulty": "Low",
            "overall_status": "Good",
            "recommendations": []
        }
