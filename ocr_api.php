<?php
// ocr_api.php
// Letakkan file ini di folder yang sama dengan ocr.html
// Path ke Python venv dan ocr_engine.py — sesuaikan jika beda

header('Content-Type: application/json');
header('Access-Control-Allow-Origin: *');

// ─── KONFIGURASI — sesuaikan path di sini ───────────────────────────────────
$PYTHON_PATH = 'D:\\AIMLPROJECT\\venv\\Scripts\\python.exe';
$OCR_ENGINE = 'D:\\AIMLPROJECT\\ocr_engine.py';
$UPLOAD_DIR = 'D:\\AIMLPROJECT\\uploads\\';
$MAX_SIZE_BYTES = 10 * 1024 * 1024; // 10MB
// ────────────────────────────────────────────────────────────────────────────

// Buat folder uploads kalau belum ada
if (!is_dir($UPLOAD_DIR)) {
    mkdir($UPLOAD_DIR, 0755, true);
}

// Cek request method
if ($_SERVER['REQUEST_METHOD'] !== 'POST') {
    echo json_encode(['success' => false, 'error' => 'Method not allowed']);
    exit;
}

// Cek file ada
if (!isset($_FILES['image']) || $_FILES['image']['error'] !== UPLOAD_ERR_OK) {
    echo json_encode(['success' => false, 'error' => 'No image uploaded']);
    exit;
}

$file = $_FILES['image'];

// Validasi ukuran
if ($file['size'] > $MAX_SIZE_BYTES) {
    echo json_encode(['success' => false, 'error' => 'File too large (max 10MB)']);
    exit;
}

// Validasi tipe file
$allowed_types = ['image/png', 'image/jpeg', 'image/jpg', 'image/webp'];
$finfo = finfo_open(FILEINFO_MIME_TYPE);
$mime  = finfo_file($finfo, $file['tmp_name']);
finfo_close($finfo);

if (!in_array($mime, $allowed_types)) {
    echo json_encode(['success' => false, 'error' => 'Invalid file type. Use PNG, JPG, or WEBP']);
    exit;
}

// Simpan file sementara dengan nama unik
$ext       = pathinfo($file['name'], PATHINFO_EXTENSION);
$filename  = uniqid('ocr_', true) . '.' . $ext;
$filepath  = $UPLOAD_DIR . $filename;

if (!move_uploaded_file($file['tmp_name'], $filepath)) {
    echo json_encode(['success' => false, 'error' => 'Failed to save uploaded file']);
    exit;
}

// Jalankan ocr_engine.py
$command = escapeshellcmd('"' . $PYTHON_PATH . '" "' . $OCR_ENGINE . '" "' . $filepath . '"');
$output  = shell_exec($command . ' 2>&1');

// Hapus file temporary setelah diproses
@unlink($filepath);

// Bersihkan output (hapus warning TensorFlow)
$lines = explode("\n", trim($output));
$clean_lines = array_filter($lines, function($line) {
    $line = trim($line);
    if (empty($line)) return false;
    // Buang semua baris warning/info dari TensorFlow dan sistem
    if (strpos($line, 'WARNING') !== false) return false;
    if (strpos($line, 'I0000') !== false) return false;
    if (strpos($line, 'W0000') !== false) return false;
    if (strpos($line, 'oneDNN') !== false) return false;
    if (strpos($line, 'TensorFlow') !== false) return false;
    if (strpos($line, 'cpu_feature') !== false) return false;
    if (strpos($line, 'tf_record') !== false) return false;
    return true;
});

$result_text = trim(implode("\n", $clean_lines));

if (empty($result_text)) {
    echo json_encode(['success' => false, 'error' => 'No text detected in image']);
    exit;
}

// Cek apakah output adalah error dari Python
if (strpos($result_text, 'Error') !== false || strpos($result_text, 'Traceback') !== false) {
    echo json_encode(['success' => false, 'error' => 'OCR processing failed: ' . $result_text]);
    exit;
}

echo json_encode([
    'success' => true,
    'text'    => $result_text
]);
?>