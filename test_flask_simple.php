<?php
/**
 * Simple Flask Server Test (Minimal Version)
 * Quick test to check if Flask is responding
 */

$flask_url = "http://agrishield.bccbsis.com:5001/health";

echo "<h2>Testing Flask Server: $flask_url</h2>";

$ch = curl_init();
curl_setopt($ch, CURLOPT_URL, $flask_url);
curl_setopt($ch, CURLOPT_RETURNTRANSFER, true);
curl_setopt($ch, CURLOPT_TIMEOUT, 10);
curl_setopt($ch, CURLOPT_CONNECTTIMEOUT, 5);

$response = curl_exec($ch);
$http_code = curl_getinfo($ch, CURLINFO_HTTP_CODE);
$error = curl_error($ch);
curl_close($ch);

if ($error) {
    echo "<p style='color: red;'><strong>❌ ERROR:</strong> $error</p>";
} else {
    echo "<p><strong>HTTP Code:</strong> $http_code</p>";
    if ($http_code == 200) {
        echo "<p style='color: green;'><strong>✅ Flask Server is ONLINE!</strong></p>";
        echo "<pre>$response</pre>";
    } else {
        echo "<p style='color: orange;'><strong>⚠️ Flask responded but with code: $http_code</strong></p>";
        echo "<pre>$response</pre>";
    }
}

// Test DNS
$domain = "agrishield.bccbsis.com";
$ip = gethostbyname($domain);
echo "<p><strong>DNS:</strong> $domain → $ip</p>";

// Test port
$connection = @fsockopen($domain, 5001, $errno, $errstr, 5);
if ($connection) {
    echo "<p style='color: green;'><strong>✅ Port 5001 is open</strong></p>";
    fclose($connection);
} else {
    echo "<p style='color: red;'><strong>❌ Port 5001 is closed</strong> ($errstr)</p>";
}
?>

