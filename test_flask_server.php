<?php
/**
 * Test Flask Server Connection
 * Upload this to your webserver and access via browser to check if Flask is running
 */

header('Content-Type: text/html; charset=utf-8');
?>
<!DOCTYPE html>
<html>
<head>
    <title>Flask Server Status Test</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 50px auto;
            padding: 20px;
            background: #f5f5f5;
        }
        .container {
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        h1 {
            color: #333;
            border-bottom: 3px solid #4CAF50;
            padding-bottom: 10px;
        }
        .test-section {
            margin: 20px 0;
            padding: 15px;
            background: #f9f9f9;
            border-left: 4px solid #2196F3;
            border-radius: 4px;
        }
        .success {
            color: #4CAF50;
            font-weight: bold;
            font-size: 18px;
        }
        .error {
            color: #f44336;
            font-weight: bold;
            font-size: 18px;
        }
        .info {
            color: #2196F3;
            margin: 10px 0;
        }
        pre {
            background: #2d2d2d;
            color: #f8f8f2;
            padding: 15px;
            border-radius: 4px;
            overflow-x: auto;
            font-size: 12px;
        }
        .status-badge {
            display: inline-block;
            padding: 5px 15px;
            border-radius: 20px;
            font-weight: bold;
            margin: 5px 0;
        }
        .status-online {
            background: #4CAF50;
            color: white;
        }
        .status-offline {
            background: #f44336;
            color: white;
        }
        button {
            background: #2196F3;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 5px;
            cursor: pointer;
            font-size: 16px;
            margin: 10px 5px;
        }
        button:hover {
            background: #1976D2;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🔍 Flask Server Status Test</h1>
        <p class="info">Testing connection to Flask ML API at: <strong>agrishield.bccbsis.com:5001</strong></p>
        
        <?php
        $flask_url = "http://agrishield.bccbsis.com:5001";
        $test_endpoints = [
            "/health" => "Health Check",
            "/" => "Root Endpoint",
            "/test_db" => "Database Test"
        ];
        
        $all_online = true;
        $results = [];
        
        foreach ($test_endpoints as $endpoint => $description) {
            $test_url = $flask_url . $endpoint;
            $result = test_flask_endpoint($test_url, $description);
            $results[] = $result;
            if (!$result['success']) {
                $all_online = false;
            }
        }
        
        // Display overall status
        echo '<div class="test-section">';
        echo '<h2>Overall Status</h2>';
        if ($all_online) {
            echo '<span class="status-badge status-online">✅ FLASK SERVER IS ONLINE</span>';
        } else {
            echo '<span class="status-badge status-offline">❌ FLASK SERVER IS OFFLINE OR UNREACHABLE</span>';
        }
        echo '</div>';
        
        // Display individual test results
        foreach ($results as $result) {
            echo '<div class="test-section">';
            echo '<h3>' . htmlspecialchars($result['description']) . '</h3>';
            echo '<p><strong>URL:</strong> <code>' . htmlspecialchars($result['url']) . '</code></p>';
            
            if ($result['success']) {
                echo '<p class="success">✅ Connection Successful</p>';
                echo '<p><strong>Response Code:</strong> ' . $result['http_code'] . '</p>';
                if (!empty($result['response'])) {
                    echo '<p><strong>Response:</strong></p>';
                    echo '<pre>' . htmlspecialchars($result['response']) . '</pre>';
                }
            } else {
                echo '<p class="error">❌ Connection Failed</p>';
                echo '<p><strong>Error:</strong> ' . htmlspecialchars($result['error']) . '</p>';
                if (!empty($result['http_code'])) {
                    echo '<p><strong>HTTP Code:</strong> ' . $result['http_code'] . '</p>';
                }
            }
            echo '</div>';
        }
        
        // Additional diagnostics
        echo '<div class="test-section">';
        echo '<h3>🔧 Diagnostics</h3>';
        echo '<p><strong>Server Time:</strong> ' . date('Y-m-d H:i:s') . '</p>';
        echo '<p><strong>PHP Version:</strong> ' . phpversion() . '</p>';
        echo '<p><strong>cURL Available:</strong> ' . (function_exists('curl_init') ? '✅ Yes' : '❌ No') . '</p>';
        
        // Test DNS resolution
        $domain = "agrishield.bccbsis.com";
        $ip = gethostbyname($domain);
        if ($ip === $domain) {
            echo '<p class="error"><strong>DNS Resolution:</strong> ❌ Failed - Cannot resolve domain</p>';
        } else {
            echo '<p class="success"><strong>DNS Resolution:</strong> ✅ ' . $domain . ' → ' . $ip . '</p>';
        }
        
        // Test port connectivity (if possible)
        echo '<p><strong>Port 5001 Test:</strong> ';
        $connection = @fsockopen($domain, 5001, $errno, $errstr, 5);
        if ($connection) {
            echo '<span class="success">✅ Port 5001 is open</span>';
            fclose($connection);
        } else {
            echo '<span class="error">❌ Port 5001 is closed or unreachable</span>';
            if ($errno > 0) {
                echo ' (Error: ' . $errstr . ')';
            }
        }
        echo '</p>';
        echo '</div>';
        
        // Troubleshooting tips
        if (!$all_online) {
            echo '<div class="test-section">';
            echo '<h3>💡 Troubleshooting Tips</h3>';
            echo '<ul>';
            echo '<li>Check if Flask server is running: <code>ps aux | grep gunicorn</code> or <code>ps aux | grep python</code></li>';
            echo '<li>Check if port 5001 is listening: <code>netstat -tulpn | grep 5001</code></li>';
            echo '<li>Verify firewall allows port 5001: <code>sudo ufw status</code></li>';
            echo '<li>Test from server command line: <code>curl http://localhost:5001/health</code></li>';
            echo '<li>Check Flask logs for errors</li>';
            echo '<li>Ensure gunicorn_config.py has <code>bind = "0.0.0.0:5001"</code> (not 127.0.0.1)</li>';
            echo '</ul>';
            echo '</div>';
        }
        ?>
        
        <div style="text-align: center; margin-top: 30px;">
            <button onclick="location.reload()">🔄 Refresh Test</button>
        </div>
    </div>
</body>
</html>

<?php
function test_flask_endpoint($url, $description) {
    $result = [
        'url' => $url,
        'description' => $description,
        'success' => false,
        'http_code' => null,
        'response' => '',
        'error' => ''
    ];
    
    if (!function_exists('curl_init')) {
        $result['error'] = 'cURL is not available on this server';
        return $result;
    }
    
    $ch = curl_init();
    curl_setopt($ch, CURLOPT_URL, $url);
    curl_setopt($ch, CURLOPT_RETURNTRANSFER, true);
    curl_setopt($ch, CURLOPT_TIMEOUT, 10);
    curl_setopt($ch, CURLOPT_CONNECTTIMEOUT, 5);
    curl_setopt($ch, CURLOPT_FOLLOWLOCATION, true);
    curl_setopt($ch, CURLOPT_SSL_VERIFYPEER, false);
    curl_setopt($ch, CURLOPT_USERAGENT, 'Flask-Test-Script/1.0');
    
    $response = curl_exec($ch);
    $http_code = curl_getinfo($ch, CURLINFO_HTTP_CODE);
    $error = curl_error($ch);
    
    curl_close($ch);
    
    if ($error) {
        $result['error'] = $error;
    } else {
        $result['http_code'] = $http_code;
        $result['response'] = $response;
        
        // Consider success if we get any HTTP response (even 404 is better than timeout)
        if ($http_code > 0) {
            $result['success'] = true;
        } else {
            $result['error'] = 'No HTTP response received';
        }
    }
    
    return $result;
}
?>

