<?php

header("Content-Type: application/json");

const DATA_FILE = __DIR__ . "/notes_data.json";
const MAX_TITLE_LEN = 200;
const MAX_TEXT_LEN = 20000; // ~20rb karakter, longgar untuk catatan panjang

function read_notes(): array {
    if (!file_exists(DATA_FILE)) {
        return [];
    }
    $raw = file_get_contents(DATA_FILE);
    $data = json_decode($raw, true);
    return is_array($data) ? $data : [];
}

function write_notes(array $notes): bool {
    // file locking (LOCK_EX) supaya aman kalau ada 2 request nyimpen bersamaan
    $fp = fopen(DATA_FILE, "c+");
    if (!$fp) return false;
    $ok = false;
    if (flock($fp, LOCK_EX)) {
        ftruncate($fp, 0);
        rewind($fp);
        $ok = fwrite($fp, json_encode($notes, JSON_PRETTY_PRINT | JSON_UNESCAPED_UNICODE)) !== false;
        flock($fp, LOCK_UN);
    }
    fclose($fp);
    return $ok;
}

function respond_error(string $message, int $code = 400): void {
    http_response_code($code);
    echo json_encode(["success" => false, "error" => $message]);
    exit;
}

function respond_success(array $payload): void {
    echo json_encode(array_merge(["success" => true], $payload));
    exit;
}

$method = $_SERVER["REQUEST_METHOD"];

// ── GET ?action=list ────────────────────────────────────────────────
if ($method === "GET") {
    $action = $_GET["action"] ?? "list";
    if ($action !== "list") {
        respond_error("Unknown action.");
    }
    $notes = read_notes();
    // terbaru duluan
    usort($notes, fn($a, $b) => strcmp($b["updated_at"] ?? "", $a["updated_at"] ?? ""));
    respond_success(["notes" => $notes]);
}

// ── POST action=save | delete ───────────────────────────────────────
if ($method === "POST") {
    $body = json_decode(file_get_contents("php://input"), true);
    if (!is_array($body)) {
        respond_error("Invalid request body.");
    }
    $action = $body["action"] ?? "";

    if ($action === "save") {
        $title = trim((string)($body["title"] ?? ""));
        $text  = trim((string)($body["text"] ?? ""));
        $id    = $body["id"] ?? null;

        if ($text === "") {
            respond_error("Note text cannot be empty.");
        }
        if (mb_strlen($title) > MAX_TITLE_LEN) {
            $title = mb_substr($title, 0, MAX_TITLE_LEN);
        }
        if (mb_strlen($text) > MAX_TEXT_LEN) {
            respond_error("Note is too long (max " . MAX_TEXT_LEN . " characters).");
        }
        if ($title === "") {
            $title = "Untitled note";
        }

        $notes = read_notes();
        $now = date("c");

        if ($id !== null) {
            // update note yang sudah ada
            $found = false;
            foreach ($notes as &$n) {
                if ($n["id"] === $id) {
                    $n["title"] = $title;
                    $n["text"] = $text;
                    $n["updated_at"] = $now;
                    $found = true;
                    $saved_note = $n;
                    break;
                }
            }
            unset($n);
            if (!$found) {
                respond_error("Note not found.", 404);
            }
        } else {
            // note baru
            $id = bin2hex(random_bytes(8));
            $saved_note = [
                "id" => $id,
                "title" => $title,
                "text" => $text,
                "created_at" => $now,
                "updated_at" => $now,
            ];
            $notes[] = $saved_note;
        }

        if (!write_notes($notes)) {
            respond_error("Could not write note to disk. Check folder permissions.", 500);
        }
        respond_success(["note" => $saved_note]);
    }

    if ($action === "delete") {
        $id = $body["id"] ?? null;
        if (!$id) {
            respond_error("Missing note id.");
        }
        $notes = read_notes();
        $remaining = array_values(array_filter($notes, fn($n) => $n["id"] !== $id));
        if (count($remaining) === count($notes)) {
            respond_error("Note not found.", 404);
        }
        if (!write_notes($remaining)) {
            respond_error("Could not update notes on disk.", 500);
        }
        respond_success(["deleted_id" => $id]);
    }

    respond_error("Unknown action.");
}

respond_error("Method not allowed.", 405);