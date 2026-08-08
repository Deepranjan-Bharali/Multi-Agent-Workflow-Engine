// gateway/main.go
package main

import (
    "bytes"
    "context"
    "crypto/sha256"
    "encoding/hex"
    "encoding/json"
    "fmt"
    "io"
    "log"
    "net/http"
    "time"

    "github.com/gofiber/fiber/v2"
    "github.com/redis/go-redis/v9"
)

var rdb *redis.Client
var ctx = context.Background()

// The internal address of your Python AI Engine
const FastAPIURL = "http://127.0.0.1:8000/api/v1/workflows/execute"

type WorkflowRequest struct {
    ThreadID  string `json:"thread_id"`
    Prompt    string `json:"prompt"`
    ProjectID int    `json:"project_id"`
}

func main() {
    // Connect to local Redis
    rdb = redis.NewClient(&redis.Options{
        Addr:     "localhost:6379",
        Password: "", 
        DB:       0,
    })

    app := fiber.New()

    // Route all workflow traffic through this middleware
    app.Post("/api/v1/workflows/execute", executeHandler)

    fmt.Println("🚀 Go Gateway running on port 3000")
    log.Fatal(app.Listen(":3000"))
}

func executeHandler(c *fiber.Ctx) error {
    var req WorkflowRequest
    if err := c.BodyParser(&req); err != nil {
        return c.Status(400).JSON(fiber.Map{"error": "Invalid request body"})
    }

    // 1. Create a deterministic cache key. 
    // We hash ONLY the ProjectID and Prompt. If 10 different users ask the 
    // exact same question, users 2-10 get the cached answer instantly.
    hashInput := fmt.Sprintf("%d|%s", req.ProjectID, req.Prompt)
    hash := sha256.Sum256([]byte(hashInput))
    cacheKey := "workflow_cache:" + hex.EncodeToString(hash[:])

    // 2. Check Redis for a Cache Hit
    cachedRes, err := rdb.Get(ctx, cacheKey).Result()
    if err == nil {
        c.Set("X-Cache", "HIT") // Custom header so you can see it working
        c.Set("Content-Type", "application/json")
        return c.SendString(cachedRes)
    }

    // 3. Cache Miss. Forward the request to FastAPI
    fastApiReqBody, _ := json.Marshal(req)
    httpReq, _ := http.NewRequest("POST", FastAPIURL, bytes.NewBuffer(fastApiReqBody))
    httpReq.Header.Set("Content-Type", "application/json")

    client := &http.Client{Timeout: 60 * time.Second}
    httpRes, err := client.Do(httpReq)
    if err != nil {
        return c.Status(500).JSON(fiber.Map{"error": "FastAPI engine unreachable"})
    }
    defer httpRes.Body.Close()

    bodyBytes, _ := io.ReadAll(httpRes.Body)

    // 4. Save the successful response to Redis (TTL: 1 Hour)
    if httpRes.StatusCode == 200 {
        rdb.Set(ctx, cacheKey, bodyBytes, 1*time.Hour)
    }

    // 5. Return the fresh response
    c.Set("X-Cache", "MISS")
    c.Set("Content-Type", "application/json")
    return c.Send(bodyBytes)
}